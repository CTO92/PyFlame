#include <gtest/gtest.h>
#include "pyflame/ir/graph.hpp"

using namespace pyflame;
using namespace pyflame::ir;

TEST(GraphTest, CreateConstant) {
    Graph graph;

    TensorSpec spec({3, 4}, DType::Float32);
    auto node = graph.create_constant(spec, nullptr, "const1");

    EXPECT_EQ(node->id(), 0);
    EXPECT_EQ(node->type(), NodeType::CONSTANT);
    EXPECT_EQ(node->name(), "const1");
    EXPECT_EQ(node->shape()[0], 3);
    EXPECT_EQ(node->shape()[1], 4);
    EXPECT_EQ(node->dtype(), DType::Float32);
}

TEST(GraphTest, CreateInput) {
    Graph graph;

    TensorSpec spec({10}, DType::Float32);
    auto node = graph.create_input(spec, "input1");

    EXPECT_EQ(node->type(), NodeType::INPUT);
    EXPECT_TRUE(node->is_input());
    EXPECT_EQ(graph.inputs().size(), 1);
    EXPECT_EQ(graph.inputs()[0], node);
}

TEST(GraphTest, CreateOperation) {
    Graph graph;

    TensorSpec spec({10}, DType::Float32);
    auto a = graph.create_input(spec, "a");
    auto b = graph.create_input(spec, "b");

    auto add = graph.create_op(OpType::ADD, {a, b}, spec, "add");

    EXPECT_EQ(add->type(), NodeType::OPERATION);
    EXPECT_EQ(add->op_type(), OpType::ADD);
    EXPECT_EQ(add->inputs().size(), 2);
    EXPECT_EQ(add->inputs()[0], a);
    EXPECT_EQ(add->inputs()[1], b);
}

TEST(GraphTest, TopologicalOrder) {
    Graph graph;

    TensorSpec spec({10}, DType::Float32);
    auto a = graph.create_input(spec, "a");
    auto b = graph.create_input(spec, "b");
    auto c = graph.create_op(OpType::ADD, {a, b}, spec, "c");
    auto d = graph.create_op(OpType::RELU, {c}, spec, "d");

    graph.mark_output(d);

    auto topo = graph.topological_order();

    EXPECT_EQ(topo.size(), 4);
    // a and b should come before c, c should come before d
    auto find_idx = [&topo](std::shared_ptr<Node> node) {
        for (size_t i = 0; i < topo.size(); ++i) {
            if (topo[i] == node) return static_cast<int>(i);
        }
        return -1;
    };

    EXPECT_LT(find_idx(a), find_idx(c));
    EXPECT_LT(find_idx(b), find_idx(c));
    EXPECT_LT(find_idx(c), find_idx(d));
}

TEST(GraphTest, NumNodes) {
    Graph graph;

    TensorSpec spec({10}, DType::Float32);
    auto a = graph.create_input(spec);
    auto b = graph.create_input(spec);
    auto c = graph.create_op(OpType::ADD, {a, b}, spec);

    EXPECT_EQ(graph.num_nodes(), 3);
    EXPECT_EQ(graph.num_ops(), 1);
}

TEST(GraphTest, NodeAttributes) {
    Graph graph;

    TensorSpec spec({10}, DType::Float32);
    auto node = graph.create_input(spec);

    node->set_attr("test_int", 42);
    node->set_attr("test_str", std::string("hello"));

    EXPECT_EQ(node->get_attr<int>("test_int"), 42);
    EXPECT_EQ(node->get_attr<std::string>("test_str"), "hello");
    EXPECT_TRUE(node->has_attr("test_int"));
    EXPECT_FALSE(node->has_attr("nonexistent"));
}

TEST(GraphTest, MarkOutput) {
    Graph graph;

    TensorSpec spec({10}, DType::Float32);
    auto a = graph.create_input(spec);
    auto b = graph.create_op(OpType::RELU, {a}, spec);

    EXPECT_EQ(graph.outputs().size(), 0);

    graph.mark_output(b);

    EXPECT_EQ(graph.outputs().size(), 1);
    EXPECT_EQ(graph.outputs()[0], b);
}

TEST(GraphTest, ToString) {
    Graph graph;

    TensorSpec spec({10}, DType::Float32);
    auto a = graph.create_input(spec, "input");
    auto b = graph.create_op(OpType::RELU, {a}, spec, "relu");

    std::string s = graph.to_string();

    EXPECT_TRUE(s.find("Graph") != std::string::npos);
    EXPECT_TRUE(s.find("input") != std::string::npos);
    EXPECT_TRUE(s.find("relu") != std::string::npos);
}
