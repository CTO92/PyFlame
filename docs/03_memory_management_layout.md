# Memory Management and Layout Transformation System

**PyFlame Version:** Pre-Release Alpha 1.0
**Document Version:** 1.0
**Last Updated:** January 10, 2026
**Status:** Design Phase

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs and designs described here are subject to change.

---

## 1. Overview

Memory management for the Cerebras WSE is fundamentally different from traditional GPU/CPU systems. The WSE is a **distributed memory architecture** where:

- Each PE has **48KB of private SRAM** (no shared memory)
- There is **no global memory** on the chip
- Data moves via **wavelets** (32-bit messages) between PEs
- Memory bandwidth within a PE is extremely high (~20 PB/s aggregate)

This document describes how PyFlame manages memory allocation, tensor layouts across the PE mesh, and layout transformations.

---

## 2. Memory Hierarchy

### 2.1 WSE Memory Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Cerebras WSE Chip                                │
│                                                                         │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     (850,000+ PEs)           │
│   │ PE  │ │ PE  │ │ PE  │ │ PE  │ │ PE  │ ...                          │
│   │48KB │ │48KB │ │48KB │ │48KB │ │48KB │                              │
│   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                              │
│      │       │       │       │       │                                  │
│      └───────┴───────┴───────┴───────┘                                  │
│                    Wavelet Fabric                                       │
│              (2D mesh interconnect)                                     │
│                         │                                               │
└─────────────────────────┼───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Host Memory                                      │
│   - Input data staging                                                  │
│   - Output data collection                                              │
│   - Unlimited capacity                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Memory Budget Per PE

| Category | Typical Allocation | Notes |
|----------|-------------------|-------|
| **Total SRAM** | 48 KB | Fixed hardware limit |
| **Code/Text** | 4-8 KB | CSL program code |
| **Stack** | 2-4 KB | Task execution stack |
| **Routing Tables** | 1-2 KB | Color/wavelet configuration |
| **Available for Data** | ~32-38 KB | Tensor tiles + scratch |

### 2.3 Memory Pressure Considerations

- **No Virtual Memory**: If data doesn't fit, compilation fails
- **No Swapping**: Everything must be resident during execution
- **Static Allocation**: Memory is allocated at compile time
- **Tiling Required**: Large tensors must be split across PEs

---

## 3. Tensor Layouts

### 3.1 Layout Abstraction

A `MeshLayout` describes how a tensor is distributed across PEs:

```cpp
// include/pyflame/core/layout.hpp
#pragma once

#include <vector>
#include <cstdint>

namespace pyflame {

// How tensor dimensions map to the PE grid
enum class LayoutType {
    SINGLE_PE,        // All data on one PE
    ROW_PARTITION,    // Split along rows of PE grid
    COL_PARTITION,    // Split along columns of PE grid
    GRID,             // 2D tiling across PE grid
    BLOCK_CYCLIC,     // Block-cyclic distribution
    CUSTOM,           // User-defined mapping
};

// Complete specification of tensor distribution
struct MeshLayout {
    LayoutType type;

    // Dimensions of PE grid used
    int32_t pe_rows;
    int32_t pe_cols;

    // For BLOCK_CYCLIC: block sizes
    int32_t block_row_size = 0;
    int32_t block_col_size = 0;

    // Which tensor dimension maps to which PE dimension
    // dim_to_pe[0] = 0 means tensor dim 0 maps to PE row
    // dim_to_pe[1] = 1 means tensor dim 1 maps to PE col
    std::vector<int> dim_to_pe;

    // Factory methods
    static MeshLayout SinglePE();
    static MeshLayout RowPartition(int32_t num_pes);
    static MeshLayout ColPartition(int32_t num_pes);
    static MeshLayout Grid(int32_t rows, int32_t cols);
    static MeshLayout BlockCyclic(int32_t rows, int32_t cols,
                                  int32_t block_rows, int32_t block_cols);

    // Compute which PE owns a given element
    PECoord pe_for_element(const std::vector<int64_t>& indices,
                           const std::vector<int64_t>& tensor_shape) const;

    // Compute local indices within a PE for a global element
    std::vector<int64_t> local_indices(const std::vector<int64_t>& global_indices,
                                       const std::vector<int64_t>& tensor_shape) const;

    // Compute tile shape for a given PE
    std::vector<int64_t> tile_shape(PECoord pe,
                                    const std::vector<int64_t>& tensor_shape) const;

    // Total number of PEs used
    int32_t total_pes() const { return pe_rows * pe_cols; }

    // Check if two layouts are compatible (can operate without communication)
    bool compatible_with(const MeshLayout& other) const;

    // Compute memory required per PE
    size_t memory_per_pe(const std::vector<int64_t>& tensor_shape,
                         size_t element_size) const;
};

}  // namespace pyflame
```

### 3.2 Layout Examples

#### Single PE Layout
```
Tensor shape: [1024, 1024]
Layout: SinglePE

┌─────────────────┐
│      PE(0,0)    │
│  [1024, 1024]   │
│   4 MB data     │
└─────────────────┘

⚠️ WARNING: 4MB > 48KB per PE!
This layout would fail for this tensor size.
```

#### Row Partition Layout
```
Tensor shape: [1024, 1024]
Layout: RowPartition(4)

PE(0,0): rows 0-255     [256, 1024]  = 1 MB per PE
PE(1,0): rows 256-511   [256, 1024]
PE(2,0): rows 512-767   [256, 1024]
PE(3,0): rows 768-1023  [256, 1024]

Still too large! Need more PEs or Grid layout.
```

#### Grid Layout (2D Tiling)
```
Tensor shape: [1024, 1024]
Layout: Grid(32, 32) = 1024 PEs

Each PE gets: [32, 32] tile = 4 KB
Total: 1024 PEs × 4 KB = 4 MB ✓

┌────┬────┬────┬────┬─────┐
│0,0 │0,1 │0,2 │... │0,31 │
├────┼────┼────┼────┼─────┤
│1,0 │1,1 │1,2 │... │1,31 │
├────┼────┼────┼────┼─────┤
│... │... │... │... │ ... │
├────┼────┼────┼────┼─────┤
│31,0│31,1│31,2│... │31,31│
└────┴────┴────┴────┴─────┘
```

### 3.3 Layout Selection Heuristics

```cpp
// include/pyflame/layout/layout_planner.hpp
#pragma once

#include "pyflame/core/layout.hpp"
#include "pyflame/ir/graph.hpp"

namespace pyflame::layout {

struct LayoutConstraints {
    size_t max_memory_per_pe = 32 * 1024;  // 32 KB available
    size_t min_tile_elements = 64;          // Minimum granularity
    int32_t max_pe_rows = 750;              // WSE-2 dimensions
    int32_t max_pe_cols = 994;
    bool prefer_square_tiles = true;
};

class LayoutPlanner {
public:
    explicit LayoutPlanner(LayoutConstraints constraints = {});

    // Choose optimal layout for a tensor
    MeshLayout plan_layout(const std::vector<int64_t>& shape,
                          DType dtype) const;

    // Choose layouts for an entire graph (coordinated)
    std::map<ir::NodeId, MeshLayout> plan_graph_layouts(
        const ir::Graph& graph
    ) const;

    // Check if a layout is valid
    bool validate_layout(const MeshLayout& layout,
                        const std::vector<int64_t>& shape,
                        DType dtype) const;

private:
    LayoutConstraints constraints_;

    // Find minimum PEs needed for a tensor
    int32_t min_pes_required(const std::vector<int64_t>& shape,
                            size_t element_size) const;

    // Score a layout (lower is better)
    double score_layout(const MeshLayout& layout,
                       const std::vector<int64_t>& shape) const;
};

}  // namespace pyflame::layout
```

### 3.4 Layout Planning Algorithm

```cpp
// src/layout/layout_planner.cpp
#include "pyflame/layout/layout_planner.hpp"
#include <cmath>

namespace pyflame::layout {

MeshLayout LayoutPlanner::plan_layout(
    const std::vector<int64_t>& shape,
    DType dtype
) const {
    size_t element_size = dtype_size(dtype);
    size_t total_bytes = numel(shape) * element_size;

    // Case 1: Fits on single PE
    if (total_bytes <= constraints_.max_memory_per_pe) {
        return MeshLayout::SinglePE();
    }

    // Case 2: Need multiple PEs
    int32_t min_pes = min_pes_required(shape, element_size);

    // For 2D tensors, prefer grid layout
    if (shape.size() == 2) {
        // Find grid dimensions that minimize communication
        int32_t best_rows = 1, best_cols = min_pes;
        double best_score = std::numeric_limits<double>::max();

        for (int32_t rows = 1; rows <= std::sqrt(min_pes) + 1; ++rows) {
            if (min_pes % rows != 0) continue;
            int32_t cols = min_pes / rows;

            MeshLayout candidate = MeshLayout::Grid(rows, cols);
            double score = score_layout(candidate, shape);

            if (score < best_score) {
                best_score = score;
                best_rows = rows;
                best_cols = cols;
            }
        }

        return MeshLayout::Grid(best_rows, best_cols);
    }

    // For 1D tensors, use row partition
    if (shape.size() == 1) {
        return MeshLayout::RowPartition(min_pes);
    }

    // For higher-D tensors, collapse to 2D and use grid
    // (Implementation detail: flatten last dims)
    return MeshLayout::Grid(
        std::min((int32_t)shape[0], constraints_.max_pe_rows),
        min_pes / std::min((int32_t)shape[0], constraints_.max_pe_rows)
    );
}

double LayoutPlanner::score_layout(
    const MeshLayout& layout,
    const std::vector<int64_t>& shape
) const {
    // Factors in scoring:
    // 1. Memory utilization per PE (want high)
    // 2. Tile aspect ratio (want close to 1:1 for matmul)
    // 3. Communication cost (fewer, larger transfers better)

    auto tile = layout.tile_shape({0, 0}, shape);
    double aspect_ratio = (double)tile[0] / (double)tile[1];

    // Penalize extreme aspect ratios
    double aspect_penalty = std::abs(std::log(aspect_ratio));

    // Penalize underutilization
    size_t tile_bytes = numel(tile) * sizeof(float);
    double utilization = (double)tile_bytes / constraints_.max_memory_per_pe;
    double util_penalty = 1.0 - utilization;

    return aspect_penalty + util_penalty;
}

}  // namespace pyflame::layout
```

---

## 4. Memory Allocation

### 4.1 Host Memory Allocator

For staging data before transfer to WSE:

```cpp
// include/pyflame/memory/host_allocator.hpp
#pragma once

#include <cstddef>
#include <memory>

namespace pyflame::memory {

// Aligned allocation for efficient DMA transfers
class HostAllocator {
public:
    static constexpr size_t DEFAULT_ALIGNMENT = 64;  // Cache line

    // Allocate aligned memory
    static void* allocate(size_t bytes, size_t alignment = DEFAULT_ALIGNMENT);

    // Deallocate
    static void deallocate(void* ptr);

    // Allocate and zero-initialize
    static void* allocate_zeroed(size_t bytes, size_t alignment = DEFAULT_ALIGNMENT);

    // RAII wrapper
    template<typename T>
    struct Deleter {
        void operator()(T* ptr) { deallocate(ptr); }
    };

    template<typename T>
    using UniquePtr = std::unique_ptr<T[], Deleter<T>>;

    template<typename T>
    static UniquePtr<T> allocate_unique(size_t count);
};

// Memory pool for frequently allocated sizes
class HostMemoryPool {
public:
    explicit HostMemoryPool(size_t block_size, size_t initial_blocks = 16);
    ~HostMemoryPool();

    void* acquire();
    void release(void* ptr);

    size_t block_size() const { return block_size_; }
    size_t total_blocks() const { return total_blocks_; }
    size_t available_blocks() const { return free_list_.size(); }

private:
    size_t block_size_;
    size_t total_blocks_;
    std::vector<void*> all_blocks_;
    std::vector<void*> free_list_;
    std::mutex mutex_;
};

}  // namespace pyflame::memory
```

### 4.2 PE Memory Planning

Memory for each PE is planned at compile time:

```cpp
// include/pyflame/memory/pe_memory_planner.hpp
#pragma once

#include <map>
#include "pyflame/ir/graph.hpp"
#include "pyflame/core/layout.hpp"

namespace pyflame::memory {

// Represents a memory region within a PE
struct MemoryRegion {
    size_t offset;      // Byte offset from PE memory base
    size_t size;        // Size in bytes
    std::string name;   // Symbolic name for debugging

    bool overlaps(const MemoryRegion& other) const {
        return !(offset + size <= other.offset || other.offset + other.size <= offset);
    }
};

// Memory allocation for a single PE
struct PEMemoryPlan {
    PECoord pe;
    size_t total_used;
    size_t total_available;
    std::vector<MemoryRegion> regions;

    // Tensor ID -> region mapping
    std::map<ir::NodeId, size_t> tensor_regions;  // region index

    bool fits() const { return total_used <= total_available; }
    size_t free_bytes() const { return total_available - total_used; }
};

// Complete memory plan for all PEs
struct MemoryPlan {
    std::map<PECoord, PEMemoryPlan> pe_plans;
    bool valid;
    std::string error_message;

    // Statistics
    size_t total_memory_used() const;
    size_t max_pe_memory_used() const;
    double average_utilization() const;
};

class PEMemoryPlanner {
public:
    static constexpr size_t PE_MEMORY_SIZE = 48 * 1024;      // 48 KB
    static constexpr size_t RESERVED_FOR_CODE = 8 * 1024;    // 8 KB
    static constexpr size_t RESERVED_FOR_STACK = 4 * 1024;   // 4 KB

    static constexpr size_t AVAILABLE_FOR_DATA =
        PE_MEMORY_SIZE - RESERVED_FOR_CODE - RESERVED_FOR_STACK;

    // Plan memory for a computation graph
    MemoryPlan plan(
        const ir::Graph& graph,
        const std::map<ir::NodeId, MeshLayout>& layouts
    );

private:
    // Liveness analysis: which tensors are live at each point
    struct LivenessInfo {
        std::map<ir::NodeId, int> first_use;   // First op that reads this tensor
        std::map<ir::NodeId, int> last_use;    // Last op that reads this tensor
    };

    LivenessInfo analyze_liveness(const ir::Graph& graph);

    // Allocate memory for one PE
    PEMemoryPlan plan_pe(
        PECoord pe,
        const std::vector<std::pair<ir::NodeId, size_t>>& allocations,
        const LivenessInfo& liveness
    );
};

}  // namespace pyflame::memory
```

### 4.3 Memory Planning Algorithm

```cpp
// src/memory/pe_memory_planner.cpp
#include "pyflame/memory/pe_memory_planner.hpp"
#include <algorithm>

namespace pyflame::memory {

MemoryPlan PEMemoryPlanner::plan(
    const ir::Graph& graph,
    const std::map<ir::NodeId, MeshLayout>& layouts
) {
    MemoryPlan result;
    result.valid = true;

    // Step 1: Analyze tensor liveness
    LivenessInfo liveness = analyze_liveness(graph);

    // Step 2: Collect all PEs and their allocations
    std::map<PECoord, std::vector<std::pair<ir::NodeId, size_t>>> pe_allocations;

    for (const auto& node : graph.all_nodes()) {
        if (node->type() == ir::NodeType::CONSTANT ||
            node->type() == ir::NodeType::INPUT ||
            node->type() == ir::NodeType::INTERMEDIATE) {

            auto layout_it = layouts.find(node->id());
            if (layout_it == layouts.end()) continue;

            const MeshLayout& layout = layout_it->second;
            const auto& spec = node->output_spec();

            // Compute tile size for each PE
            for (int r = 0; r < layout.pe_rows; ++r) {
                for (int c = 0; c < layout.pe_cols; ++c) {
                    PECoord pe{r, c};
                    auto tile_shape = layout.tile_shape(pe, spec.shape);
                    size_t tile_bytes = numel(tile_shape) * dtype_size(spec.dtype);

                    pe_allocations[pe].push_back({node->id(), tile_bytes});
                }
            }
        }
    }

    // Step 3: Plan memory for each PE
    for (auto& [pe, allocations] : pe_allocations) {
        PEMemoryPlan pe_plan = plan_pe(pe, allocations, liveness);

        if (!pe_plan.fits()) {
            result.valid = false;
            result.error_message = "PE (" + std::to_string(pe.row) + ", " +
                                   std::to_string(pe.col) + ") exceeds memory: " +
                                   std::to_string(pe_plan.total_used) + " > " +
                                   std::to_string(pe_plan.total_available);
            return result;
        }

        result.pe_plans[pe] = std::move(pe_plan);
    }

    return result;
}

PEMemoryPlan PEMemoryPlanner::plan_pe(
    PECoord pe,
    const std::vector<std::pair<ir::NodeId, size_t>>& allocations,
    const LivenessInfo& liveness
) {
    PEMemoryPlan plan;
    plan.pe = pe;
    plan.total_available = AVAILABLE_FOR_DATA;
    plan.total_used = 0;

    // Simple bump allocator (can be improved with register allocation algorithms)
    size_t current_offset = 0;

    for (const auto& [node_id, size] : allocations) {
        // Align to 4 bytes
        current_offset = (current_offset + 3) & ~3;

        MemoryRegion region;
        region.offset = current_offset;
        region.size = size;
        region.name = "tensor_" + std::to_string(node_id);

        plan.tensor_regions[node_id] = plan.regions.size();
        plan.regions.push_back(region);

        current_offset += size;
    }

    plan.total_used = current_offset;
    return plan;
}

LivenessInfo PEMemoryPlanner::analyze_liveness(const ir::Graph& graph) {
    LivenessInfo info;

    auto topo_order = graph.topological_order();

    for (int op_idx = 0; op_idx < topo_order.size(); ++op_idx) {
        const auto& node = topo_order[op_idx];

        // Record first use of each input
        for (const auto& input : node->inputs()) {
            if (info.first_use.find(input->id()) == info.first_use.end()) {
                info.first_use[input->id()] = op_idx;
            }
            info.last_use[input->id()] = op_idx;
        }

        // The output is "created" at this op
        info.first_use[node->id()] = op_idx;
    }

    // Outputs are live until the end
    for (const auto& output : graph.outputs()) {
        info.last_use[output->id()] = topo_order.size() - 1;
    }

    return info;
}

}  // namespace pyflame::memory
```

---

## 5. Layout Transformations

### 5.1 When Transformations Are Needed

Layout transformations are required when:

1. **Operation Mismatch**: An operation requires different input layouts
2. **User Request**: User explicitly calls `to_layout()`
3. **Optimization**: A different layout would be more efficient

```cpp
// Example: Matmul requires specific layouts
Tensor A = randn({1024, 512}, RowPartition(4));
Tensor B = randn({512, 256}, ColPartition(4));
Tensor C = matmul(A, B);  // Requires layout transformation!

// For efficient matmul:
// - A should be row-partitioned (each PE has full rows)
// - B should be col-partitioned (each PE has full cols)
// - OR both should be Grid layout with matching dimensions
```

### 5.2 Transformation Types

```cpp
// include/pyflame/layout/layout_transform.hpp
#pragma once

#include "pyflame/core/layout.hpp"
#include "pyflame/ir/graph.hpp"

namespace pyflame::layout {

enum class TransformType {
    IDENTITY,           // No transformation needed
    REDISTRIBUTE,       // Same data, different PE assignment
    REPLICATE,          // Copy data to all PEs
    GATHER,             // Collect distributed data to one PE
    SCATTER,            // Distribute from one PE to many
    TRANSPOSE,          // Change dimension mapping
    RESHARD,            // General redistribution (most expensive)
};

// Description of a layout transformation
struct LayoutTransform {
    TransformType type;
    MeshLayout from_layout;
    MeshLayout to_layout;
    std::vector<int64_t> tensor_shape;

    // Estimated cost (in wavelet messages)
    size_t estimated_cost() const;

    // Generate communication pattern
    struct CommunicationPattern {
        struct Transfer {
            PECoord source;
            PECoord dest;
            size_t start_offset;  // Within source tile
            size_t size;          // Bytes to transfer
        };
        std::vector<Transfer> transfers;
    };

    CommunicationPattern generate_pattern() const;
};

class LayoutTransformer {
public:
    // Check if transformation is needed
    static bool needs_transform(const MeshLayout& from, const MeshLayout& to);

    // Create transformation descriptor
    static LayoutTransform create_transform(
        const MeshLayout& from,
        const MeshLayout& to,
        const std::vector<int64_t>& shape
    );

    // Insert transformation nodes into graph
    static void insert_transforms(
        ir::Graph& graph,
        const std::map<ir::NodeId, MeshLayout>& target_layouts
    );

private:
    // Specific transformation implementations
    static LayoutTransform create_gather(const MeshLayout& from,
                                        const std::vector<int64_t>& shape);
    static LayoutTransform create_scatter(const MeshLayout& to,
                                         const std::vector<int64_t>& shape);
    static LayoutTransform create_reshard(const MeshLayout& from,
                                         const MeshLayout& to,
                                         const std::vector<int64_t>& shape);
};

}  // namespace pyflame::layout
```

### 5.3 Transformation Cost Model

```cpp
// src/layout/layout_transform.cpp
#include "pyflame/layout/layout_transform.hpp"

namespace pyflame::layout {

size_t LayoutTransform::estimated_cost() const {
    switch (type) {
        case TransformType::IDENTITY:
            return 0;

        case TransformType::REDISTRIBUTE: {
            // Cost: data size * average hop distance
            size_t data_bytes = numel(tensor_shape) * sizeof(float);
            int avg_hops = (from_layout.pe_rows + from_layout.pe_cols) / 2;
            return data_bytes * avg_hops;
        }

        case TransformType::REPLICATE: {
            // Cost: full copy to all PEs
            size_t data_bytes = numel(tensor_shape) * sizeof(float);
            int num_copies = to_layout.total_pes() - 1;
            return data_bytes * num_copies;
        }

        case TransformType::GATHER: {
            // Cost: all PEs send to one
            size_t tile_bytes = numel(tensor_shape) * sizeof(float) / from_layout.total_pes();
            int max_hops = from_layout.pe_rows + from_layout.pe_cols;
            return tile_bytes * from_layout.total_pes() * max_hops;
        }

        case TransformType::SCATTER: {
            // Cost: one PE sends to all
            size_t data_bytes = numel(tensor_shape) * sizeof(float);
            int max_hops = to_layout.pe_rows + to_layout.pe_cols;
            return data_bytes * max_hops;
        }

        case TransformType::TRANSPOSE:
        case TransformType::RESHARD: {
            // Worst case: all-to-all communication
            size_t data_bytes = numel(tensor_shape) * sizeof(float);
            int num_pes = std::max(from_layout.total_pes(), to_layout.total_pes());
            int diameter = std::max(
                from_layout.pe_rows + from_layout.pe_cols,
                to_layout.pe_rows + to_layout.pe_cols
            );
            return data_bytes * diameter;
        }
    }
    return 0;
}

LayoutTransform::CommunicationPattern LayoutTransform::generate_pattern() const {
    CommunicationPattern pattern;

    switch (type) {
        case TransformType::IDENTITY:
            // No transfers needed
            break;

        case TransformType::GATHER: {
            // All PEs send to PE(0,0)
            PECoord dest{0, 0};
            size_t tile_size = numel(tensor_shape) * sizeof(float) / from_layout.total_pes();

            for (int r = 0; r < from_layout.pe_rows; ++r) {
                for (int c = 0; c < from_layout.pe_cols; ++c) {
                    if (r == 0 && c == 0) continue;

                    CommunicationPattern::Transfer t;
                    t.source = {r, c};
                    t.dest = dest;
                    t.start_offset = 0;
                    t.size = tile_size;
                    pattern.transfers.push_back(t);
                }
            }
            break;
        }

        case TransformType::SCATTER: {
            // PE(0,0) sends to all PEs
            PECoord source{0, 0};
            size_t full_size = numel(tensor_shape) * sizeof(float);
            size_t tile_size = full_size / to_layout.total_pes();

            size_t offset = tile_size;  // First tile stays local
            for (int r = 0; r < to_layout.pe_rows; ++r) {
                for (int c = 0; c < to_layout.pe_cols; ++c) {
                    if (r == 0 && c == 0) continue;

                    CommunicationPattern::Transfer t;
                    t.source = source;
                    t.dest = {r, c};
                    t.start_offset = offset;
                    t.size = tile_size;
                    pattern.transfers.push_back(t);
                    offset += tile_size;
                }
            }
            break;
        }

        case TransformType::RESHARD: {
            // General case: compute overlap between old and new tiles
            // Each PE may need to send parts of its data to multiple destinations
            // (Complex implementation - simplified here)
            // ...
            break;
        }

        default:
            break;
    }

    return pattern;
}

}  // namespace pyflame::layout
```

### 5.4 Transformation Insertion Pass

```cpp
// src/layout/transform_insertion.cpp
#include "pyflame/layout/layout_transform.hpp"
#include "pyflame/ir/graph.hpp"

namespace pyflame::layout {

void LayoutTransformer::insert_transforms(
    ir::Graph& graph,
    const std::map<ir::NodeId, MeshLayout>& target_layouts
) {
    // Process operations in topological order
    auto topo_order = graph.topological_order();

    for (auto& node : topo_order) {
        if (!node->is_operation()) continue;

        // Check each input for layout mismatch
        for (size_t i = 0; i < node->inputs().size(); ++i) {
            auto input = node->inputs()[i];
            auto input_layout = target_layouts.at(input->id());

            // Determine required layout for this input
            MeshLayout required = get_required_input_layout(
                node->op_type(), i, target_layouts.at(node->id())
            );

            if (!input_layout.compatible_with(required)) {
                // Insert transformation node
                auto transform = create_transform(
                    input_layout,
                    required,
                    input->output_spec().shape
                );

                // Create new node for transformed tensor
                ir::TensorSpec transformed_spec = input->output_spec();
                transformed_spec.layout = required;

                auto transform_node = graph.create_op(
                    ir::OpType::LAYOUT_TRANSFORM,
                    {input},
                    transformed_spec,
                    "transform_" + std::to_string(input->id())
                );

                // Rewire: this input now comes from transform node
                node->replace_input(i, transform_node);

                // Store transform metadata for code generation
                transform_node->set_metadata("transform_type",
                    static_cast<int>(transform.type));
            }
        }
    }
}

MeshLayout get_required_input_layout(
    ir::OpType op,
    size_t input_idx,
    const MeshLayout& output_layout
) {
    switch (op) {
        case ir::OpType::ADD:
        case ir::OpType::MUL:
        case ir::OpType::SUB:
        case ir::OpType::DIV:
            // Elementwise ops: inputs must match output layout
            return output_layout;

        case ir::OpType::MATMUL:
            // Matmul has specific requirements:
            // Input A: partitioned along rows (or matching grid)
            // Input B: partitioned along cols (or matching grid)
            if (input_idx == 0) {
                // First input (A): row partition or grid
                if (output_layout.type == LayoutType::GRID) {
                    return MeshLayout::Grid(output_layout.pe_rows, 1);
                }
                return MeshLayout::RowPartition(output_layout.pe_rows);
            } else {
                // Second input (B): col partition or grid
                if (output_layout.type == LayoutType::GRID) {
                    return MeshLayout::Grid(1, output_layout.pe_cols);
                }
                return MeshLayout::ColPartition(output_layout.pe_cols);
            }

        case ir::OpType::SUM:
        case ir::OpType::MEAN:
        case ir::OpType::MAX:
            // Reductions: input layout determines output
            return output_layout;

        default:
            return output_layout;
    }
}

}  // namespace pyflame::layout
```

---

## 6. Data Transfer (Host <-> WSE)

### 6.1 Transfer Manager

```cpp
// include/pyflame/runtime/transfer.hpp
#pragma once

#include <memory>
#include "pyflame/core/tensor.hpp"
#include "pyflame/core/layout.hpp"

namespace pyflame::runtime {

// Handles data movement between host memory and WSE
class TransferManager {
public:
    // Transfer tensor data from host to WSE
    void host_to_device(
        const Tensor& tensor,
        const void* host_data,
        size_t size_bytes
    );

    // Transfer tensor data from WSE to host
    void device_to_host(
        const Tensor& tensor,
        void* host_data,
        size_t size_bytes
    );

    // Async transfers
    struct TransferHandle {
        uint64_t id;
        bool is_complete() const;
        void wait() const;
    };

    TransferHandle host_to_device_async(
        const Tensor& tensor,
        const void* host_data,
        size_t size_bytes
    );

    TransferHandle device_to_host_async(
        const Tensor& tensor,
        void* host_data,
        size_t size_bytes
    );

    // Statistics
    size_t total_bytes_transferred() const;
    double average_transfer_bandwidth() const;

private:
    // Implementation uses Cerebras SDK's memcpy module
    // ...
};

}  // namespace pyflame::runtime
```

### 6.2 Transfer with Layout Awareness

```cpp
// src/runtime/transfer.cpp
#include "pyflame/runtime/transfer.hpp"

namespace pyflame::runtime {

void TransferManager::host_to_device(
    const Tensor& tensor,
    const void* host_data,
    size_t size_bytes
) {
    const MeshLayout& layout = tensor.layout();
    const auto& shape = tensor.shape();
    size_t element_size = dtype_size(tensor.dtype());

    if (layout.type == LayoutType::SINGLE_PE) {
        // Simple case: transfer entire tensor to one PE
        transfer_to_pe({0, 0}, host_data, size_bytes);

    } else if (layout.type == LayoutType::GRID) {
        // Distribute tiles to appropriate PEs
        const auto* src = static_cast<const uint8_t*>(host_data);

        for (int r = 0; r < layout.pe_rows; ++r) {
            for (int c = 0; c < layout.pe_cols; ++c) {
                PECoord pe{r, c};
                auto tile_shape = layout.tile_shape(pe, shape);
                size_t tile_bytes = numel(tile_shape) * element_size;

                // Compute source offset in host tensor
                // (Assumes row-major host layout)
                size_t src_offset = compute_tile_offset(r, c, layout, shape, element_size);

                transfer_to_pe(pe, src + src_offset, tile_bytes);
            }
        }

    } else {
        // Other layouts...
        // Similar logic for ROW_PARTITION, COL_PARTITION, etc.
    }
}

void TransferManager::device_to_host(
    const Tensor& tensor,
    void* host_data,
    size_t size_bytes
) {
    const MeshLayout& layout = tensor.layout();
    const auto& shape = tensor.shape();
    size_t element_size = dtype_size(tensor.dtype());

    if (layout.type == LayoutType::SINGLE_PE) {
        // Simple case: read entire tensor from one PE
        transfer_from_pe({0, 0}, host_data, size_bytes);

    } else if (layout.type == LayoutType::GRID) {
        // Gather tiles from all PEs
        auto* dst = static_cast<uint8_t*>(host_data);

        for (int r = 0; r < layout.pe_rows; ++r) {
            for (int c = 0; c < layout.pe_cols; ++c) {
                PECoord pe{r, c};
                auto tile_shape = layout.tile_shape(pe, shape);
                size_t tile_bytes = numel(tile_shape) * element_size;

                size_t dst_offset = compute_tile_offset(r, c, layout, shape, element_size);

                transfer_from_pe(pe, dst + dst_offset, tile_bytes);
            }
        }
    }
}

}  // namespace pyflame::runtime
```

---

## 7. Optimization Strategies

### 7.1 Layout Propagation

Propagate compatible layouts through the graph to minimize transformations:

```cpp
// Algorithm: Backward layout propagation
//
// 1. Start from output nodes with fixed layouts
// 2. Work backward through the graph
// 3. For each operation:
//    a. If output layout is fixed, propagate compatible layouts to inputs
//    b. If multiple compatible layouts exist, choose the one that
//       minimizes total transformation cost
//
// Example:
//   c = matmul(a, b)  with c: Grid(4, 4)
//   Propagate: a -> RowPartition(4), b -> ColPartition(4)
//   If a was Grid(2, 8), insert transformation
```

### 7.2 Memory Reuse

Reuse memory for tensors with non-overlapping lifetimes:

```cpp
// Algorithm: Linear scan register allocation (adapted for PE memory)
//
// 1. Sort allocations by first use
// 2. Maintain free list of memory regions
// 3. For each allocation:
//    a. Check if any free region fits
//    b. If yes, reuse that region
//    c. If no, allocate new region
// 4. On last use of a tensor, add its region to free list
//
// This can reduce memory pressure by 30-50% in typical models
```

### 7.3 Double Buffering

For streaming computations, use double buffering:

```cpp
// Double buffering scheme:
//
// PE Memory:
// [Buffer A: 16KB][Buffer B: 16KB][Compute scratch: 4KB]
//
// Timeline:
// T0: Load data into Buffer A, previous compute on Buffer B
// T1: Compute on Buffer A, load next data into Buffer B
// T2: Compute on Buffer B, load next data into Buffer A
// ...
//
// This hides transfer latency behind computation
```

---

## 8. Python API for Layout Control

```python
# python/pyflame/layout.py
"""Layout control API for PyFlame tensors."""

import pyflame._pyflame_cpp as _cpp

class Layout:
    """Tensor distribution layout across the PE mesh."""

    @staticmethod
    def single_pe():
        """All data on a single PE.

        Best for small tensors (< 32KB).
        """
        return _cpp.MeshLayout.SinglePE()

    @staticmethod
    def row_partition(num_pes):
        """Partition tensor rows across PEs.

        Each PE gets tensor_rows / num_pes rows.

        Args:
            num_pes: Number of PEs to use.
        """
        return _cpp.MeshLayout.RowPartition(num_pes)

    @staticmethod
    def col_partition(num_pes):
        """Partition tensor columns across PEs.

        Each PE gets tensor_cols / num_pes columns.

        Args:
            num_pes: Number of PEs to use.
        """
        return _cpp.MeshLayout.ColPartition(num_pes)

    @staticmethod
    def grid(rows, cols):
        """2D tiling across a PE grid.

        Tensor is divided into rows × cols tiles, one per PE.

        Args:
            rows: Number of PE rows.
            cols: Number of PE columns.
        """
        return _cpp.MeshLayout.Grid(rows, cols)

    @staticmethod
    def auto(tensor_shape, dtype='float32'):
        """Automatically choose optimal layout.

        Uses heuristics to select layout based on tensor size
        and memory constraints.

        Args:
            tensor_shape: Shape of the tensor.
            dtype: Data type.

        Returns:
            Optimal MeshLayout for this tensor.
        """
        return _cpp.LayoutPlanner().plan_layout(list(tensor_shape), dtype)


# Convenience functions
def to_layout(tensor, layout):
    """Convert tensor to a new layout.

    May involve data redistribution across PEs.

    Args:
        tensor: Input tensor.
        layout: Target MeshLayout.

    Returns:
        Tensor with new layout (may share data if compatible).
    """
    if tensor.layout == layout:
        return tensor
    return tensor.to_layout(layout)


def get_tile_shape(tensor, pe_coord=(0, 0)):
    """Get the shape of tensor tile on a specific PE.

    Args:
        tensor: Distributed tensor.
        pe_coord: (row, col) of PE.

    Returns:
        Tuple of tile dimensions.
    """
    return tuple(tensor.layout.tile_shape(pe_coord, tensor.shape))


def memory_usage_per_pe(tensor):
    """Calculate memory usage per PE for a tensor.

    Args:
        tensor: Tensor to analyze.

    Returns:
        Dict with 'min', 'max', 'average' bytes per PE.
    """
    layout = tensor.layout
    shape = tensor.shape
    dtype = tensor.dtype

    elem_size = {'float32': 4, 'float16': 2, 'bfloat16': 2, 'int32': 4, 'int16': 2}[str(dtype)]

    usages = []
    for r in range(layout.pe_rows):
        for c in range(layout.pe_cols):
            tile = layout.tile_shape((r, c), shape)
            usages.append(prod(tile) * elem_size)

    return {
        'min': min(usages),
        'max': max(usages),
        'average': sum(usages) / len(usages),
        'total': sum(usages),
    }
```

---

## 9. Example: Memory-Aware Model Design

```python
import pyflame as pf
from pyflame.layout import Layout, memory_usage_per_pe

# Design consideration: 48KB per PE, ~32KB available for data
MAX_PE_MEMORY = 32 * 1024  # 32 KB

def create_linear_layer(in_features, out_features, batch_size):
    """Create a linear layer with automatic layout selection."""

    # Weight: [in_features, out_features]
    weight_bytes = in_features * out_features * 4  # float32

    # Choose layout based on weight size
    if weight_bytes <= MAX_PE_MEMORY:
        # Small: single PE
        weight_layout = Layout.single_pe()
    else:
        # Large: grid layout
        # Heuristic: aim for ~16KB tiles (leave room for activations)
        target_tile_bytes = 16 * 1024
        target_tile_elements = target_tile_bytes // 4

        # Find grid dimensions
        import math
        total_elements = in_features * out_features
        num_tiles = math.ceil(total_elements / target_tile_elements)
        grid_rows = int(math.sqrt(num_tiles))
        grid_cols = math.ceil(num_tiles / grid_rows)

        weight_layout = Layout.grid(grid_rows, grid_cols)

    print(f"Linear({in_features}, {out_features}):")
    print(f"  Weight layout: {weight_layout}")
    print(f"  PEs used: {weight_layout.pe_rows * weight_layout.pe_cols}")

    weights = pf.randn([in_features, out_features], layout=weight_layout)
    bias = pf.zeros([out_features])

    print(f"  Memory per PE: {memory_usage_per_pe(weights)}")

    def forward(x):
        return pf.relu(x @ weights + bias)

    return forward


# Example: Building a small MLP
layer1 = create_linear_layer(1024, 512, batch_size=32)
layer2 = create_linear_layer(512, 256, batch_size=32)
layer3 = create_linear_layer(256, 10, batch_size=32)

# Forward pass
x = pf.randn([32, 1024])
h1 = layer1(x)
h2 = layer2(h1)
output = layer3(h2)

# Evaluate
result = pf.eval(output)
print(f"Output shape: {result.shape}")
```

---

## 10. Future Work

### 10.1 Planned Enhancements

1. **Automatic Layout Optimization**: ML-based layout selection
2. **Memory Compression**: On-the-fly compression for large tensors
3. **Prefetching**: Anticipate data needs and prefetch via wavelets
4. **Checkpointing**: Save/restore intermediate tensors for gradient checkpointing

### 10.2 Research Areas

1. **Sparse Tensor Layouts**: Efficient storage for sparse data
2. **Heterogeneous Layouts**: Different dtypes in different regions
3. **Dynamic Reshaping**: Efficient layout changes during execution

---

*Document Version: 1.0*
*Authors: PyFlame Team*
*Last Updated: January 10, 2026*
