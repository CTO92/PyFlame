// PyFlame Python bindings using pybind11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "pyflame/core/tensor.hpp"
#include "pyflame/core/tensor_impl.hpp"
#include "pyflame/core/dtype.hpp"
#include "pyflame/core/layout.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/ir/node.hpp"
#include "pyflame/backend/csl_codegen.hpp"

namespace py = pybind11;
using namespace pyflame;

// Helper to convert numpy array to Tensor
Tensor numpy_to_tensor(py::array_t<float> arr, DType dtype, MeshLayout layout) {
    py::buffer_info buf = arr.request();

    std::vector<int64_t> shape;
    for (auto dim : buf.shape) {
        shape.push_back(static_cast<int64_t>(dim));
    }

    return Tensor::from_data(buf.ptr, shape, dtype, layout);
}

// Helper to convert Tensor to numpy array
py::array_t<float> tensor_to_numpy(Tensor& t) {
    // Force evaluation
    t.eval();

    auto shape = t.shape();
    std::vector<ssize_t> py_shape(shape.begin(), shape.end());

    // Create numpy array and copy data
    py::array_t<float> result(py_shape);
    py::buffer_info buf = result.request();

    std::memcpy(buf.ptr, t.data<float>(), t.numel() * sizeof(float));

    return result;
}

PYBIND11_MODULE(_pyflame_cpp, m) {
    m.doc() = "PyFlame: Native deep learning framework for Cerebras WSE";

    // ========================================================================
    // DType enum
    // ========================================================================
    py::enum_<DType>(m, "DType")
        .value("float32", DType::Float32)
        .value("float16", DType::Float16)
        .value("bfloat16", DType::BFloat16)
        .value("int32", DType::Int32)
        .value("int16", DType::Int16)
        .value("int8", DType::Int8)
        .value("bool_", DType::Bool)
        .export_values();

    m.def("dtype_size", &dtype_size, "Get size in bytes of a dtype");
    m.def("dtype_name", &dtype_name, "Get name of a dtype");

    // ========================================================================
    // PECoord
    // ========================================================================
    py::class_<PECoord>(m, "PECoord")
        .def(py::init<>())
        .def(py::init<int32_t, int32_t>(), py::arg("row"), py::arg("col"))
        .def_readwrite("row", &PECoord::row)
        .def_readwrite("col", &PECoord::col)
        .def("__repr__", &PECoord::to_string)
        .def("__eq__", &PECoord::operator==)
        .def("__hash__", [](const PECoord& c) {
            return std::hash<int64_t>()(static_cast<int64_t>(c.row) << 32 | c.col);
        });

    // ========================================================================
    // MeshLayout
    // ========================================================================
    py::class_<MeshLayout>(m, "MeshLayout")
        .def(py::init<>())
        .def_static("single_pe", &MeshLayout::SinglePE)
        .def_static("row_partition", &MeshLayout::RowPartition)
        .def_static("col_partition", &MeshLayout::ColPartition)
        .def_static("grid", &MeshLayout::Grid)
        .def_static("replicated", &MeshLayout::Replicated)
        .def_readonly("type", &MeshLayout::type)
        .def_readonly("pe_rows", &MeshLayout::pe_rows)
        .def_readonly("pe_cols", &MeshLayout::pe_cols)
        .def("total_pes", &MeshLayout::total_pes)
        .def("tile_shape", &MeshLayout::tile_shape)
        .def("compatible_with", &MeshLayout::compatible_with)
        .def("memory_per_pe", &MeshLayout::memory_per_pe)
        .def("__repr__", &MeshLayout::to_string)
        .def("__eq__", &MeshLayout::operator==);

    // ========================================================================
    // Tensor class
    // ========================================================================
    py::class_<Tensor>(m, "Tensor")
        // Static factory methods
        .def_static("zeros", &Tensor::zeros,
            py::arg("shape"),
            py::arg("dtype") = DType::Float32,
            py::arg("layout") = MeshLayout::SinglePE(),
            "Create a tensor filled with zeros")

        .def_static("ones", &Tensor::ones,
            py::arg("shape"),
            py::arg("dtype") = DType::Float32,
            py::arg("layout") = MeshLayout::SinglePE(),
            "Create a tensor filled with ones")

        .def_static("full", &Tensor::full,
            py::arg("shape"),
            py::arg("value"),
            py::arg("dtype") = DType::Float32,
            py::arg("layout") = MeshLayout::SinglePE(),
            "Create a tensor filled with a value")

        .def_static("randn", &Tensor::randn,
            py::arg("shape"),
            py::arg("dtype") = DType::Float32,
            py::arg("layout") = MeshLayout::SinglePE(),
            "Create a tensor with random normal values")

        .def_static("rand", &Tensor::rand,
            py::arg("shape"),
            py::arg("dtype") = DType::Float32,
            py::arg("layout") = MeshLayout::SinglePE(),
            "Create a tensor with random uniform values in [0, 1)")

        .def_static("arange", &Tensor::arange,
            py::arg("start"),
            py::arg("end"),
            py::arg("step") = 1,
            py::arg("dtype") = DType::Float32,
            "Create a tensor with values from start to end")

        .def_static("from_numpy", [](py::array_t<float> arr) {
            return numpy_to_tensor(arr, DType::Float32, MeshLayout::SinglePE());
        }, py::arg("arr"), "Create a tensor from a numpy array")

        // Properties
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("layout", &Tensor::layout)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("is_scalar", &Tensor::is_scalar)

        .def("size", &Tensor::size, py::arg("dim"))

        // Evaluation
        .def("eval", &Tensor::eval, "Force evaluation of this tensor")
        .def("is_evaluated", &Tensor::is_evaluated)

        // NumPy conversion
        .def("numpy", [](Tensor& t) {
            return tensor_to_numpy(t);
        }, "Convert to numpy array (forces evaluation)")

        // Reshape operations
        .def("view", &Tensor::view, py::arg("shape"))
        .def("reshape", &Tensor::reshape, py::arg("shape"))
        .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
        .def("t", &Tensor::t)
        .def("squeeze", &Tensor::squeeze, py::arg("dim") = -1)
        .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim"))
        .def("contiguous", &Tensor::contiguous)

        // Arithmetic operators
        .def("__add__", [](const Tensor& a, const Tensor& b) { return a + b; })
        .def("__add__", [](const Tensor& a, float b) { return a + b; })
        .def("__radd__", [](const Tensor& a, float b) { return a + b; })

        .def("__sub__", [](const Tensor& a, const Tensor& b) { return a - b; })
        .def("__sub__", [](const Tensor& a, float b) { return a - b; })
        .def("__rsub__", [](const Tensor& a, float b) { return b - a; })

        .def("__mul__", [](const Tensor& a, const Tensor& b) { return a * b; })
        .def("__mul__", [](const Tensor& a, float b) { return a * b; })
        .def("__rmul__", [](const Tensor& a, float b) { return a * b; })

        .def("__truediv__", [](const Tensor& a, const Tensor& b) { return a / b; })
        .def("__truediv__", [](const Tensor& a, float b) { return a / b; })

        .def("__neg__", [](const Tensor& a) { return -a; })

        .def("__matmul__", [](const Tensor& a, const Tensor& b) { return matmul(a, b); })

        // Reductions
        .def("sum", &Tensor::sum,
            py::arg("dim") = py::none(),
            py::arg("keepdim") = false)
        .def("mean", &Tensor::mean,
            py::arg("dim") = py::none(),
            py::arg("keepdim") = false)
        .def("max", &Tensor::max,
            py::arg("dim") = py::none(),
            py::arg("keepdim") = false)
        .def("min", &Tensor::min,
            py::arg("dim") = py::none(),
            py::arg("keepdim") = false)

        // Comparisons
        .def("__eq__", [](const Tensor& a, const Tensor& b) { return a == b; })
        .def("__ne__", [](const Tensor& a, const Tensor& b) { return a != b; })
        .def("__lt__", [](const Tensor& a, const Tensor& b) { return a < b; })
        .def("__le__", [](const Tensor& a, const Tensor& b) { return a <= b; })
        .def("__gt__", [](const Tensor& a, const Tensor& b) { return a > b; })
        .def("__ge__", [](const Tensor& a, const Tensor& b) { return a >= b; })

        // String representation
        .def("__repr__", &Tensor::to_string);

    // ========================================================================
    // Free functions
    // ========================================================================

    // Matrix operations
    m.def("matmul", &matmul, py::arg("a"), py::arg("b"),
        "Matrix multiplication");

    // Activation functions
    m.def("relu", &relu, py::arg("x"), "ReLU activation");
    m.def("sigmoid", &sigmoid, py::arg("x"), "Sigmoid activation");
    m.def("tanh", static_cast<Tensor(*)(const Tensor&)>(&pyflame::tanh),
        py::arg("x"), "Tanh activation");
    m.def("gelu", &gelu, py::arg("x"), "GELU activation");
    m.def("silu", &silu, py::arg("x"), "SiLU (Swish) activation");
    m.def("softmax", &softmax, py::arg("x"), py::arg("dim") = -1, "Softmax");
    m.def("log_softmax", &log_softmax, py::arg("x"), py::arg("dim") = -1, "Log-Softmax");

    // Elementwise math
    m.def("abs", static_cast<Tensor(*)(const Tensor&)>(&pyflame::abs),
        py::arg("x"), "Absolute value");
    m.def("sqrt", static_cast<Tensor(*)(const Tensor&)>(&pyflame::sqrt),
        py::arg("x"), "Square root");
    m.def("exp", static_cast<Tensor(*)(const Tensor&)>(&pyflame::exp),
        py::arg("x"), "Exponential");
    m.def("log", static_cast<Tensor(*)(const Tensor&)>(&pyflame::log),
        py::arg("x"), "Natural logarithm");
    m.def("sin", static_cast<Tensor(*)(const Tensor&)>(&pyflame::sin),
        py::arg("x"), "Sine");
    m.def("cos", static_cast<Tensor(*)(const Tensor&)>(&pyflame::cos),
        py::arg("x"), "Cosine");

    // Tensor combination
    m.def("cat", &cat, py::arg("tensors"), py::arg("dim") = 0,
        "Concatenate tensors");
    m.def("stack", &stack, py::arg("tensors"), py::arg("dim") = 0,
        "Stack tensors");

    // ========================================================================
    // Graph access (for advanced users)
    // ========================================================================

    py::class_<ir::TensorSpec>(m, "TensorSpec")
        .def_readonly("shape", &ir::TensorSpec::shape)
        .def_readonly("dtype", &ir::TensorSpec::dtype)
        .def_readonly("layout", &ir::TensorSpec::layout)
        .def("numel", &ir::TensorSpec::numel)
        .def("size_bytes", &ir::TensorSpec::size_bytes)
        .def("ndim", &ir::TensorSpec::ndim)
        .def("__repr__", &ir::TensorSpec::to_string);

    py::class_<ir::Node, std::shared_ptr<ir::Node>>(m, "Node")
        .def("id", &ir::Node::id)
        .def("name", &ir::Node::name)
        .def("output_spec", &ir::Node::output_spec)
        .def("shape", &ir::Node::shape)
        .def("dtype", &ir::Node::dtype)
        .def("layout", &ir::Node::layout)
        .def("is_constant", &ir::Node::is_constant)
        .def("is_input", &ir::Node::is_input)
        .def("is_operation", &ir::Node::is_operation)
        .def("__repr__", &ir::Node::to_string);

    py::class_<ir::Graph, std::shared_ptr<ir::Graph>>(m, "Graph")
        .def(py::init<>())
        .def("num_nodes", &ir::Graph::num_nodes)
        .def("num_ops", &ir::Graph::num_ops)
        .def("estimated_memory_bytes", &ir::Graph::estimated_memory_bytes)
        .def("topological_order", &ir::Graph::topological_order)
        .def("__repr__", &ir::Graph::to_string);

    // Access graph from tensor
    m.def("get_graph", [](const Tensor& t) {
        return t.graph();
    }, py::arg("tensor"), "Get the computation graph for a tensor");

    m.def("get_node", [](const Tensor& t) {
        return t.node();
    }, py::arg("tensor"), "Get the graph node for a tensor");

    // ========================================================================
    // CSL Code Generation
    // ========================================================================

    py::class_<backend::CodeGenOptions>(m, "CodeGenOptions")
        .def(py::init<>())
        .def_readwrite("output_dir", &backend::CodeGenOptions::output_dir)
        .def_readwrite("target", &backend::CodeGenOptions::target)
        .def_readwrite("optimize", &backend::CodeGenOptions::optimize)
        .def_readwrite("generate_debug_info", &backend::CodeGenOptions::generate_debug_info)
        .def_readwrite("fabric_width", &backend::CodeGenOptions::fabric_width)
        .def_readwrite("fabric_height", &backend::CodeGenOptions::fabric_height);

    py::class_<backend::CodeGenResult>(m, "CodeGenResult")
        .def_readonly("success", &backend::CodeGenResult::success)
        .def_readonly("error_message", &backend::CodeGenResult::error_message)
        .def_readonly("output_dir", &backend::CodeGenResult::output_dir)
        .def_readonly("generated_files", &backend::CodeGenResult::generated_files)
        .def_readonly("sources", &backend::CodeGenResult::sources);

    py::class_<backend::CSLCodeGenerator>(m, "CSLCodeGenerator")
        .def(py::init<>())
        .def("generate", &backend::CSLCodeGenerator::generate,
            py::arg("graph"),
            py::arg("options") = backend::CodeGenOptions{})
        .def("generate_source", &backend::CSLCodeGenerator::generate_source,
            py::arg("graph"),
            py::arg("options") = backend::CodeGenOptions{});

    // Convenience function
    m.def("compile_to_csl", [](const Tensor& t, const std::string& output_dir) {
        auto graph = t.graph();
        if (!graph) {
            throw std::runtime_error("Tensor has no associated graph");
        }
        graph->mark_output(t.node());

        backend::CSLCodeGenerator codegen;
        backend::CodeGenOptions opts;
        opts.output_dir = output_dir;

        return codegen.generate(*graph, opts);
    }, py::arg("tensor"), py::arg("output_dir") = "./pyflame_csl_output",
    "Compile tensor computation graph to CSL");

    // ========================================================================
    // Version info
    // ========================================================================
    m.attr("__version__") = "0.1.0";
}
