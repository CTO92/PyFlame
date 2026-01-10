#include "pyflame/core/tensor_impl.hpp"
#include "pyflame/ir/graph.hpp"

namespace pyflame {

// Thread-local storage for the current computation graph
static thread_local std::shared_ptr<ir::Graph> g_current_graph;

std::shared_ptr<ir::Graph> TensorImpl::get_current_graph() {
    if (!g_current_graph) {
        g_current_graph = std::make_shared<ir::Graph>();
    }
    return g_current_graph;
}

void TensorImpl::set_current_graph(std::shared_ptr<ir::Graph> graph) {
    g_current_graph = graph;
}

}  // namespace pyflame
