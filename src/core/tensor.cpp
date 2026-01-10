#include "pyflame/core/tensor.hpp"
#include "pyflame/core/tensor_impl.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/ir/shape_inference.hpp"

#include <sstream>
#include <iomanip>
#include <algorithm>

namespace pyflame {

// ============================================================================
// Constructors and assignment
// ============================================================================

Tensor::Tensor() : impl_(nullptr) {}

Tensor::Tensor(const Tensor& other) : impl_(other.impl_) {}

Tensor::Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}

Tensor& Tensor::operator=(const Tensor& other) {
    impl_ = other.impl_;
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}

Tensor::~Tensor() = default;

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

// ============================================================================
// Factory methods
// ============================================================================

Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype, MeshLayout layout) {
    return Tensor(TensorImpl::zeros(shape, dtype, layout));
}

Tensor Tensor::ones(std::vector<int64_t> shape, DType dtype, MeshLayout layout) {
    return Tensor(TensorImpl::ones(shape, dtype, layout));
}

Tensor Tensor::full(std::vector<int64_t> shape, float value, DType dtype, MeshLayout layout) {
    return Tensor(TensorImpl::full(shape, value, dtype, layout));
}

Tensor Tensor::randn(std::vector<int64_t> shape, DType dtype, MeshLayout layout) {
    return Tensor(TensorImpl::randn(shape, dtype, layout));
}

Tensor Tensor::rand(std::vector<int64_t> shape, DType dtype, MeshLayout layout) {
    return Tensor(TensorImpl::rand(shape, dtype, layout));
}

Tensor Tensor::arange(int64_t start, int64_t end, int64_t step, DType dtype) {
    return Tensor(TensorImpl::arange(start, end, step, dtype));
}

Tensor Tensor::from_data(const void* data, std::vector<int64_t> shape, DType dtype, MeshLayout layout) {
    return Tensor(TensorImpl::from_data(data, shape, dtype, layout));
}

// ============================================================================
// Properties
// ============================================================================

std::vector<int64_t> Tensor::shape() const {
    return impl_ ? impl_->shape() : std::vector<int64_t>{};
}

int64_t Tensor::size(int dim) const {
    if (!impl_) return 0;
    auto s = impl_->shape();
    if (dim < 0) dim += static_cast<int>(s.size());
    return (dim >= 0 && dim < static_cast<int>(s.size())) ? s[dim] : 0;
}

int64_t Tensor::numel() const {
    return impl_ ? impl_->numel() : 0;
}

DType Tensor::dtype() const {
    return impl_ ? impl_->dtype() : DType::Float32;
}

MeshLayout Tensor::layout() const {
    return impl_ ? impl_->layout() : MeshLayout::SinglePE();
}

int Tensor::ndim() const {
    return impl_ ? impl_->ndim() : 0;
}

bool Tensor::is_scalar() const {
    return impl_ && impl_->ndim() == 0;
}

// ============================================================================
// Data access
// ============================================================================

template<typename T>
T* Tensor::data() {
    if (!impl_) return nullptr;
    impl_->materialize();
    return static_cast<T*>(impl_->data());
}

template<typename T>
const T* Tensor::data() const {
    if (!impl_) return nullptr;
    const_cast<TensorImpl*>(impl_.get())->materialize();
    return static_cast<const T*>(impl_->data());
}

void* Tensor::data_ptr() {
    if (!impl_) return nullptr;
    return impl_->materialize();
}

const void* Tensor::data_ptr() const {
    if (!impl_) return nullptr;
    return const_cast<TensorImpl*>(impl_.get())->materialize();
}

// Explicit instantiations
template float* Tensor::data<float>();
template const float* Tensor::data<float>() const;
template int32_t* Tensor::data<int32_t>();
template const int32_t* Tensor::data<int32_t>() const;
template int16_t* Tensor::data<int16_t>();
template const int16_t* Tensor::data<int16_t>() const;

// ============================================================================
// Evaluation
// ============================================================================

bool Tensor::is_evaluated() const {
    return impl_ && impl_->is_evaluated();
}

Tensor& Tensor::eval() {
    if (impl_) {
        impl_->materialize();
    }
    return *this;
}

void Tensor::eval_all(std::vector<Tensor>& tensors) {
    for (auto& t : tensors) {
        t.eval();
    }
}

// ============================================================================
// Graph access
// ============================================================================

std::shared_ptr<ir::Graph> Tensor::graph() const {
    return impl_ ? impl_->graph() : nullptr;
}

std::shared_ptr<ir::Node> Tensor::node() const {
    return impl_ ? impl_->node() : nullptr;
}

// ============================================================================
// Reshape operations
// ============================================================================

Tensor Tensor::view(std::vector<int64_t> new_shape) const {
    if (!impl_) return Tensor();
    return Tensor(impl_->apply_reshape(new_shape));
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    return view(new_shape);
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (!impl_) return Tensor();
    return Tensor(impl_->apply_transpose(dim0, dim1));
}

Tensor Tensor::t() const {
    if (!impl_ || impl_->ndim() != 2) {
        throw std::runtime_error("t() expects a 2D tensor");
    }
    return transpose(0, 1);
}

Tensor Tensor::squeeze(int dim) const {
    if (!impl_) return Tensor();
    auto s = impl_->shape();

    std::vector<int64_t> new_shape;
    if (dim == -1) {
        // Remove all dimensions of size 1
        for (auto d : s) {
            if (d != 1) new_shape.push_back(d);
        }
    } else {
        if (dim < 0) dim += static_cast<int>(s.size());
        for (int i = 0; i < static_cast<int>(s.size()); ++i) {
            if (i == dim && s[i] == 1) continue;
            new_shape.push_back(s[i]);
        }
    }

    return reshape(new_shape);
}

Tensor Tensor::unsqueeze(int dim) const {
    if (!impl_) return Tensor();
    auto s = impl_->shape();

    if (dim < 0) dim += static_cast<int>(s.size()) + 1;

    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(s.size()) + 1; ++i) {
        if (i == dim) {
            new_shape.push_back(1);
        }
        if (i < static_cast<int>(s.size())) {
            new_shape.push_back(s[i]);
        }
    }

    return reshape(new_shape);
}

Tensor Tensor::contiguous() const {
    // For now, all tensors are contiguous
    return *this;
}

// ============================================================================
// Slicing
// ============================================================================

Tensor Tensor::slice(int dim, int64_t start, int64_t end) const {
    // Simplified slice - would need proper implementation
    throw std::runtime_error("slice not yet implemented");
}

Tensor Tensor::operator[](int64_t idx) const {
    // Simplified indexing - would need proper implementation
    throw std::runtime_error("indexing not yet implemented");
}

// ============================================================================
// Layout operations
// ============================================================================

Tensor Tensor::to_layout(MeshLayout new_layout) const {
    if (!impl_) return Tensor();
    // For now, just create a view with new layout
    // Full implementation would insert layout transform nodes
    return *this;
}

Tensor Tensor::to(DType new_dtype) const {
    if (!impl_) return Tensor();
    if (impl_->dtype() == new_dtype) return *this;
    // Type conversion would need implementation
    throw std::runtime_error("dtype conversion not yet implemented");
}

// ============================================================================
// Arithmetic operators
// ============================================================================

Tensor Tensor::operator+(const Tensor& other) const {
    if (!impl_ || !other.impl_) return Tensor();
    return Tensor(impl_->apply_binary(ir::OpType::ADD, other.impl_));
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (!impl_ || !other.impl_) return Tensor();
    return Tensor(impl_->apply_binary(ir::OpType::SUB, other.impl_));
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (!impl_ || !other.impl_) return Tensor();
    return Tensor(impl_->apply_binary(ir::OpType::MUL, other.impl_));
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (!impl_ || !other.impl_) return Tensor();
    return Tensor(impl_->apply_binary(ir::OpType::DIV, other.impl_));
}

Tensor Tensor::operator-() const {
    if (!impl_) return Tensor();
    return Tensor(impl_->apply_unary(ir::OpType::NEG));
}

Tensor Tensor::operator+(float scalar) const {
    return *this + Tensor::full(shape(), scalar, dtype());
}

Tensor Tensor::operator-(float scalar) const {
    return *this - Tensor::full(shape(), scalar, dtype());
}

Tensor Tensor::operator*(float scalar) const {
    return *this * Tensor::full(shape(), scalar, dtype());
}

Tensor Tensor::operator/(float scalar) const {
    return *this / Tensor::full(shape(), scalar, dtype());
}

// ============================================================================
// In-place operations
// ============================================================================

Tensor& Tensor::operator+=(const Tensor& other) {
    *this = *this + other;
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    *this = *this - other;
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    *this = *this * other;
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    *this = *this / other;
    return *this;
}

Tensor& Tensor::add_(const Tensor& other) {
    return *this += other;
}

Tensor& Tensor::sub_(const Tensor& other) {
    return *this -= other;
}

Tensor& Tensor::mul_(const Tensor& other) {
    return *this *= other;
}

Tensor& Tensor::div_(const Tensor& other) {
    return *this /= other;
}

Tensor& Tensor::fill_(float value) {
    *this = Tensor::full(shape(), value, dtype(), layout());
    return *this;
}

// ============================================================================
// Reduction operations
// ============================================================================

Tensor Tensor::sum(std::optional<int> dim, bool keepdim) const {
    if (!impl_) return Tensor();
    return Tensor(impl_->apply_reduction(ir::OpType::SUM, dim, keepdim));
}

Tensor Tensor::mean(std::optional<int> dim, bool keepdim) const {
    if (!impl_) return Tensor();
    return Tensor(impl_->apply_reduction(ir::OpType::MEAN, dim, keepdim));
}

Tensor Tensor::max(std::optional<int> dim, bool keepdim) const {
    if (!impl_) return Tensor();
    return Tensor(impl_->apply_reduction(ir::OpType::MAX, dim, keepdim));
}

Tensor Tensor::min(std::optional<int> dim, bool keepdim) const {
    if (!impl_) return Tensor();
    return Tensor(impl_->apply_reduction(ir::OpType::MIN, dim, keepdim));
}

Tensor Tensor::prod(std::optional<int> dim, bool keepdim) const {
    if (!impl_) return Tensor();
    return Tensor(impl_->apply_reduction(ir::OpType::PROD, dim, keepdim));
}

// ============================================================================
// Comparison operators
// ============================================================================

Tensor Tensor::operator==(const Tensor& other) const {
    if (!impl_ || !other.impl_) return Tensor();
    return Tensor(impl_->apply_binary(ir::OpType::EQ, other.impl_));
}

Tensor Tensor::operator!=(const Tensor& other) const {
    if (!impl_ || !other.impl_) return Tensor();
    return Tensor(impl_->apply_binary(ir::OpType::NE, other.impl_));
}

Tensor Tensor::operator<(const Tensor& other) const {
    if (!impl_ || !other.impl_) return Tensor();
    return Tensor(impl_->apply_binary(ir::OpType::LT, other.impl_));
}

Tensor Tensor::operator<=(const Tensor& other) const {
    if (!impl_ || !other.impl_) return Tensor();
    return Tensor(impl_->apply_binary(ir::OpType::LE, other.impl_));
}

Tensor Tensor::operator>(const Tensor& other) const {
    if (!impl_ || !other.impl_) return Tensor();
    return Tensor(impl_->apply_binary(ir::OpType::GT, other.impl_));
}

Tensor Tensor::operator>=(const Tensor& other) const {
    if (!impl_ || !other.impl_) return Tensor();
    return Tensor(impl_->apply_binary(ir::OpType::GE, other.impl_));
}

// ============================================================================
// String representation
// ============================================================================

std::string Tensor::to_string() const {
    if (!impl_) return "Tensor(null)";

    std::ostringstream ss;
    ss << "Tensor(";

    // Shape
    ss << "shape=[";
    auto s = shape();
    for (size_t i = 0; i < s.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << s[i];
    }
    ss << "], ";

    // Dtype
    ss << "dtype=" << dtype_name(dtype()) << ", ";

    // Layout
    ss << "layout=" << layout().to_string();

    // If evaluated and small, show values
    if (is_evaluated() && numel() <= 10) {
        ss << ", data=[";
        const float* d = data<float>();
        for (int64_t i = 0; i < numel(); ++i) {
            if (i > 0) ss << ", ";
            ss << std::fixed << std::setprecision(4) << d[i];
        }
        ss << "]";
    }

    ss << ")";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    return os << t.to_string();
}

// ============================================================================
// Free function operations
// ============================================================================

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (!a.impl() || !b.impl()) return Tensor();

    auto graph = a.graph();

    // Infer output spec
    auto output_spec = ir::infer_matmul_spec(
        a.node()->output_spec(),
        b.node()->output_spec()
    );

    // Create matmul node
    auto node = graph->create_op(
        ir::OpType::MATMUL,
        {a.node(), b.node()},
        output_spec,
        "matmul_" + std::to_string(graph->num_nodes())
    );

    return Tensor(TensorImpl::from_node(graph, node));
}

Tensor operator@(const Tensor& a, const Tensor& b) {
    return matmul(a, b);
}

// Activation functions
Tensor relu(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::RELU));
}

Tensor sigmoid(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::SIGMOID));
}

Tensor tanh(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::TANH));
}

Tensor gelu(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::GELU));
}

Tensor silu(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::SILU));
}

Tensor softmax(const Tensor& x, int dim) {
    if (!x.impl()) return Tensor();
    auto graph = x.graph();

    auto output_spec = ir::infer_softmax_spec(x.node()->output_spec(), dim);

    auto node = graph->create_op(
        ir::OpType::SOFTMAX,
        {x.node()},
        output_spec,
        "softmax_" + std::to_string(graph->num_nodes())
    );
    node->set_attr("dim", dim);

    return Tensor(TensorImpl::from_node(graph, node));
}

Tensor log_softmax(const Tensor& x, int dim) {
    if (!x.impl()) return Tensor();
    auto graph = x.graph();

    auto output_spec = ir::infer_softmax_spec(x.node()->output_spec(), dim);

    auto node = graph->create_op(
        ir::OpType::LOG_SOFTMAX,
        {x.node()},
        output_spec,
        "log_softmax_" + std::to_string(graph->num_nodes())
    );
    node->set_attr("dim", dim);

    return Tensor(TensorImpl::from_node(graph, node));
}

// Elementwise math
Tensor abs(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::ABS));
}

Tensor sqrt(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::SQRT));
}

Tensor exp(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::EXP));
}

Tensor log(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::LOG));
}

Tensor sin(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::SIN));
}

Tensor cos(const Tensor& x) {
    if (!x.impl()) return Tensor();
    return Tensor(x.impl()->apply_unary(ir::OpType::COS));
}

Tensor pow(const Tensor& base, const Tensor& exponent) {
    if (!base.impl() || !exponent.impl()) return Tensor();
    return Tensor(base.impl()->apply_binary(ir::OpType::POW, exponent.impl()));
}

Tensor pow(const Tensor& base, float exponent) {
    return pow(base, Tensor::full(base.shape(), exponent, base.dtype()));
}

// Tensor combination
Tensor cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) return Tensor();

    auto graph = tensors[0].graph();

    // Collect input specs
    std::vector<ir::TensorSpec> specs;
    std::vector<std::shared_ptr<ir::Node>> input_nodes;
    for (const auto& t : tensors) {
        specs.push_back(t.node()->output_spec());
        input_nodes.push_back(t.node());
    }

    auto output_spec = ir::infer_concat_spec(specs, dim);

    auto node = graph->create_op(
        ir::OpType::CONCAT,
        input_nodes,
        output_spec,
        "concat_" + std::to_string(graph->num_nodes())
    );
    node->set_attr("dim", dim);

    return Tensor(TensorImpl::from_node(graph, node));
}

Tensor stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) return Tensor();

    // Stack = unsqueeze each tensor, then cat
    std::vector<Tensor> unsqueezed;
    for (const auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }

    return cat(unsqueezed, dim);
}

Tensor broadcast_to(const Tensor& x, std::vector<int64_t> shape) {
    if (!x.impl()) return Tensor();
    // Simplified - just reshape if compatible
    return x.reshape(shape);
}

}  // namespace pyflame
