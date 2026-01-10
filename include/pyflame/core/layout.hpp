#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace pyflame {

/// 2D PE mesh coordinate
struct PECoord {
    int32_t row = 0;
    int32_t col = 0;

    PECoord() = default;
    PECoord(int32_t r, int32_t c) : row(r), col(c) {}

    bool operator==(const PECoord& other) const {
        return row == other.row && col == other.col;
    }

    bool operator!=(const PECoord& other) const {
        return !(*this == other);
    }

    bool operator<(const PECoord& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }

    std::string to_string() const {
        return "PE(" + std::to_string(row) + ", " + std::to_string(col) + ")";
    }
};

/// How tensor dimensions map to the PE grid
enum class LayoutType : uint8_t {
    SINGLE_PE = 0,      // All data on one PE
    ROW_PARTITION = 1,  // Split along rows of PE grid
    COL_PARTITION = 2,  // Split along columns of PE grid
    GRID = 3,           // 2D tiling across PE grid
    REPLICATED = 4,     // Data replicated on all PEs
};

/// Complete specification of tensor distribution across PEs
struct MeshLayout {
    LayoutType type = LayoutType::SINGLE_PE;
    int32_t pe_rows = 1;
    int32_t pe_cols = 1;

    // Default constructor
    MeshLayout() = default;

    // Constructor
    MeshLayout(LayoutType t, int32_t rows, int32_t cols)
        : type(t), pe_rows(rows), pe_cols(cols) {}

    /// Create a single PE layout
    static MeshLayout SinglePE() {
        return MeshLayout(LayoutType::SINGLE_PE, 1, 1);
    }

    /// Create a row-partitioned layout
    static MeshLayout RowPartition(int32_t num_pes) {
        return MeshLayout(LayoutType::ROW_PARTITION, num_pes, 1);
    }

    /// Create a column-partitioned layout
    static MeshLayout ColPartition(int32_t num_pes) {
        return MeshLayout(LayoutType::COL_PARTITION, 1, num_pes);
    }

    /// Create a 2D grid layout
    static MeshLayout Grid(int32_t rows, int32_t cols) {
        return MeshLayout(LayoutType::GRID, rows, cols);
    }

    /// Create a replicated layout
    static MeshLayout Replicated(int32_t rows, int32_t cols) {
        return MeshLayout(LayoutType::REPLICATED, rows, cols);
    }

    /// Total number of PEs used
    int32_t total_pes() const {
        return pe_rows * pe_cols;
    }

    /// Compute the tile shape for a given PE
    std::vector<int64_t> tile_shape(PECoord pe,
                                    const std::vector<int64_t>& tensor_shape) const {
        if (tensor_shape.empty()) {
            return {};
        }

        std::vector<int64_t> result = tensor_shape;

        switch (type) {
            case LayoutType::SINGLE_PE:
            case LayoutType::REPLICATED:
                // Full tensor on each PE
                return result;

            case LayoutType::ROW_PARTITION:
                if (!result.empty()) {
                    int64_t total_rows = result[0];
                    int64_t rows_per_pe = (total_rows + pe_rows - 1) / pe_rows;
                    int64_t start = pe.row * rows_per_pe;
                    int64_t end = std::min(start + rows_per_pe, total_rows);
                    result[0] = end - start;
                }
                return result;

            case LayoutType::COL_PARTITION:
                if (result.size() >= 2) {
                    int64_t total_cols = result[1];
                    int64_t cols_per_pe = (total_cols + pe_cols - 1) / pe_cols;
                    int64_t start = pe.col * cols_per_pe;
                    int64_t end = std::min(start + cols_per_pe, total_cols);
                    result[1] = end - start;
                } else if (result.size() == 1) {
                    int64_t total = result[0];
                    int64_t per_pe = (total + pe_cols - 1) / pe_cols;
                    int64_t start = pe.col * per_pe;
                    int64_t end = std::min(start + per_pe, total);
                    result[0] = end - start;
                }
                return result;

            case LayoutType::GRID:
                if (!result.empty()) {
                    int64_t total_rows = result[0];
                    int64_t rows_per_pe = (total_rows + pe_rows - 1) / pe_rows;
                    int64_t row_start = pe.row * rows_per_pe;
                    int64_t row_end = std::min(row_start + rows_per_pe, total_rows);
                    result[0] = row_end - row_start;
                }
                if (result.size() >= 2) {
                    int64_t total_cols = result[1];
                    int64_t cols_per_pe = (total_cols + pe_cols - 1) / pe_cols;
                    int64_t col_start = pe.col * cols_per_pe;
                    int64_t col_end = std::min(col_start + cols_per_pe, total_cols);
                    result[1] = col_end - col_start;
                }
                return result;

            default:
                return result;
        }
    }

    /// Check if two layouts are compatible (same distribution)
    bool compatible_with(const MeshLayout& other) const {
        return type == other.type &&
               pe_rows == other.pe_rows &&
               pe_cols == other.pe_cols;
    }

    /// Compute memory required per PE (max across all PEs)
    size_t memory_per_pe(const std::vector<int64_t>& tensor_shape,
                         size_t element_size) const {
        size_t max_bytes = 0;
        for (int32_t r = 0; r < pe_rows; ++r) {
            for (int32_t c = 0; c < pe_cols; ++c) {
                auto tile = tile_shape({r, c}, tensor_shape);
                int64_t numel = 1;
                for (auto d : tile) numel *= d;
                size_t bytes = static_cast<size_t>(numel) * element_size;
                max_bytes = std::max(max_bytes, bytes);
            }
        }
        return max_bytes;
    }

    bool operator==(const MeshLayout& other) const {
        return type == other.type &&
               pe_rows == other.pe_rows &&
               pe_cols == other.pe_cols;
    }

    bool operator!=(const MeshLayout& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        switch (type) {
            case LayoutType::SINGLE_PE:
                return "SinglePE";
            case LayoutType::ROW_PARTITION:
                return "RowPartition(" + std::to_string(pe_rows) + ")";
            case LayoutType::COL_PARTITION:
                return "ColPartition(" + std::to_string(pe_cols) + ")";
            case LayoutType::GRID:
                return "Grid(" + std::to_string(pe_rows) + ", " + std::to_string(pe_cols) + ")";
            case LayoutType::REPLICATED:
                return "Replicated(" + std::to_string(pe_rows) + ", " + std::to_string(pe_cols) + ")";
            default:
                return "Unknown";
        }
    }
};

/// Stream output for MeshLayout
inline std::ostream& operator<<(std::ostream& os, const MeshLayout& layout) {
    return os << layout.to_string();
}

}  // namespace pyflame

// Hash support for PECoord (for use in unordered_map)
namespace std {
template <>
struct hash<pyflame::PECoord> {
    size_t operator()(const pyflame::PECoord& coord) const {
        return hash<int64_t>()(static_cast<int64_t>(coord.row) << 32 | coord.col);
    }
};
}  // namespace std
