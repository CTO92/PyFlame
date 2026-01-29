#pragma once

#include <vector>
#include <memory>
#include <optional>
#include <functional>

#include "pyflame/core/dtype.hpp"
#include "pyflame/core/layout.hpp"

namespace pyflame {

// Forward declarations
class TensorImpl;

namespace ir {
    class Graph;
    class Node;
}

/// Main tensor class - a handle to a node in the computation graph
class Tensor {
public:
    // Default constructor (creates empty tensor)
    Tensor();

    // Copy and move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    // Factory methods
    static Tensor zeros(std::vector<int64_t> shape, DType dtype = DType::Float32,
                       MeshLayout layout = MeshLayout::SinglePE());
    static Tensor ones(std::vector<int64_t> shape, DType dtype = DType::Float32,
                      MeshLayout layout = MeshLayout::SinglePE());
    static Tensor full(std::vector<int64_t> shape, float value, DType dtype = DType::Float32,
                      MeshLayout layout = MeshLayout::SinglePE());
    static Tensor randn(std::vector<int64_t> shape, DType dtype = DType::Float32,
                       MeshLayout layout = MeshLayout::SinglePE());
    static Tensor rand(std::vector<int64_t> shape, DType dtype = DType::Float32,
                      MeshLayout layout = MeshLayout::SinglePE());
    static Tensor arange(int64_t start, int64_t end, int64_t step = 1,
                        DType dtype = DType::Float32);
    static Tensor from_data(const void* data, std::vector<int64_t> shape,
                           DType dtype = DType::Float32,
                           MeshLayout layout = MeshLayout::SinglePE(),
                           size_t data_size = 0);  // Optional buffer size for validation

    // Properties
    std::vector<int64_t> shape() const;
    int64_t size(int dim) const;
    int64_t numel() const;
    DType dtype() const;
    MeshLayout layout() const;
    int ndim() const;
    bool is_scalar() const;

    // Data access (forces evaluation)
    template<typename T>
    T* data();

    template<typename T>
    const T* data() const;

    void* data_ptr();
    const void* data_ptr() const;

    // Evaluation control
    bool is_evaluated() const;
    Tensor& eval();
    static void eval_all(std::vector<Tensor>& tensors);

    // Reshape operations
    Tensor view(std::vector<int64_t> new_shape) const;
    Tensor reshape(std::vector<int64_t> new_shape) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor t() const;  // 2D transpose shorthand
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor contiguous() const;
    Tensor clone() const;  // Create a copy of the tensor

    // Slicing
    Tensor slice(int dim, int64_t start, int64_t end) const;
    Tensor operator[](int64_t idx) const;

    // Layout operations
    Tensor to_layout(MeshLayout new_layout) const;
    Tensor to(DType dtype) const;

    // Arithmetic operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // In-place operations
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    Tensor& add_(const Tensor& other);
    Tensor& sub_(const Tensor& other);
    Tensor& mul_(const Tensor& other);
    Tensor& div_(const Tensor& other);
    Tensor& fill_(float value);

    // Reduction operations
    Tensor sum(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
    Tensor mean(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
    Tensor max(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
    Tensor min(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
    Tensor prod(std::optional<int> dim = std::nullopt, bool keepdim = false) const;

    // Comparison operators
    Tensor operator==(const Tensor& other) const;
    Tensor operator!=(const Tensor& other) const;
    Tensor operator<(const Tensor& other) const;
    Tensor operator<=(const Tensor& other) const;
    Tensor operator>(const Tensor& other) const;
    Tensor operator>=(const Tensor& other) const;

    // Internal access
    std::shared_ptr<TensorImpl> impl() const { return impl_; }
    std::shared_ptr<ir::Graph> graph() const;
    std::shared_ptr<ir::Node> node() const;

    // Factory method for creating Tensor from TensorImpl (for internal use)
    static Tensor from_impl(std::shared_ptr<TensorImpl> impl) { return Tensor(impl); }

    // Autograd methods
    Tensor grad() const;          // Get gradient tensor
    void zero_grad();             // Zero out gradient
    bool requires_grad() const;   // Check if gradient is tracked
    void set_requires_grad(bool requires_grad);  // Enable/disable gradient tracking

    // String representation
    std::string to_string() const;

private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    std::shared_ptr<TensorImpl> impl_;

    friend class TensorImpl;

    // Friend declarations for free functions that need private constructor access
    friend Tensor matmul(const Tensor& a, const Tensor& b);
    friend Tensor relu(const Tensor& x);
    friend Tensor sigmoid(const Tensor& x);
    friend Tensor tanh(const Tensor& x);
    friend Tensor gelu(const Tensor& x);
    friend Tensor silu(const Tensor& x);
    friend Tensor softmax(const Tensor& x, int dim);
    friend Tensor log_softmax(const Tensor& x, int dim);
    friend Tensor abs(const Tensor& x);
    friend Tensor sqrt(const Tensor& x);
    friend Tensor exp(const Tensor& x);
    friend Tensor log(const Tensor& x);
    friend Tensor sin(const Tensor& x);
    friend Tensor cos(const Tensor& x);
    friend Tensor pow(const Tensor& base, const Tensor& exponent);
    friend Tensor cat(const std::vector<Tensor>& tensors, int dim);
};

// Free function operations
Tensor matmul(const Tensor& a, const Tensor& b);
// Note: operator@ is not valid in C++, use matmul() instead

// Activation functions
Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor gelu(const Tensor& x);
Tensor silu(const Tensor& x);
Tensor softmax(const Tensor& x, int dim = -1);
Tensor log_softmax(const Tensor& x, int dim = -1);

// Elementwise math
Tensor abs(const Tensor& x);
Tensor sqrt(const Tensor& x);
Tensor exp(const Tensor& x);
Tensor log(const Tensor& x);
Tensor sin(const Tensor& x);
Tensor cos(const Tensor& x);
Tensor pow(const Tensor& base, const Tensor& exponent);
Tensor pow(const Tensor& base, float exponent);
Tensor clamp(const Tensor& x, float min_val, float max_val);
Tensor maximum(const Tensor& a, const Tensor& b);  // Element-wise max
Tensor minimum(const Tensor& a, const Tensor& b);  // Element-wise min

// Tensor combination
Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);
Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);

// Broadcasting
Tensor broadcast_to(const Tensor& x, std::vector<int64_t> shape);

// Scalar operators (reverse order)
inline Tensor operator+(float scalar, const Tensor& t) { return t + scalar; }
inline Tensor operator*(float scalar, const Tensor& t) { return t * scalar; }
inline Tensor operator-(float scalar, const Tensor& t) { return (-t) + scalar; }
inline Tensor operator/(float scalar, const Tensor& t) {
    return Tensor::full(t.shape(), scalar, t.dtype()) / t;
}

// Stream output
std::ostream& operator<<(std::ostream& os, const Tensor& t);

}  // namespace pyflame
