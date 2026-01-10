#include <gtest/gtest.h>
#include "pyflame/core/layout.hpp"

using namespace pyflame;

TEST(PECoordTest, DefaultConstruction) {
    PECoord coord;
    EXPECT_EQ(coord.row, 0);
    EXPECT_EQ(coord.col, 0);
}

TEST(PECoordTest, Construction) {
    PECoord coord(3, 5);
    EXPECT_EQ(coord.row, 3);
    EXPECT_EQ(coord.col, 5);
}

TEST(PECoordTest, Equality) {
    PECoord a(1, 2);
    PECoord b(1, 2);
    PECoord c(1, 3);

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(MeshLayoutTest, SinglePE) {
    auto layout = MeshLayout::SinglePE();
    EXPECT_EQ(layout.type, LayoutType::SINGLE_PE);
    EXPECT_EQ(layout.pe_rows, 1);
    EXPECT_EQ(layout.pe_cols, 1);
    EXPECT_EQ(layout.total_pes(), 1);
}

TEST(MeshLayoutTest, RowPartition) {
    auto layout = MeshLayout::RowPartition(4);
    EXPECT_EQ(layout.type, LayoutType::ROW_PARTITION);
    EXPECT_EQ(layout.pe_rows, 4);
    EXPECT_EQ(layout.pe_cols, 1);
    EXPECT_EQ(layout.total_pes(), 4);
}

TEST(MeshLayoutTest, ColPartition) {
    auto layout = MeshLayout::ColPartition(8);
    EXPECT_EQ(layout.type, LayoutType::COL_PARTITION);
    EXPECT_EQ(layout.pe_rows, 1);
    EXPECT_EQ(layout.pe_cols, 8);
    EXPECT_EQ(layout.total_pes(), 8);
}

TEST(MeshLayoutTest, Grid) {
    auto layout = MeshLayout::Grid(4, 8);
    EXPECT_EQ(layout.type, LayoutType::GRID);
    EXPECT_EQ(layout.pe_rows, 4);
    EXPECT_EQ(layout.pe_cols, 8);
    EXPECT_EQ(layout.total_pes(), 32);
}

TEST(MeshLayoutTest, TileShapeSinglePE) {
    auto layout = MeshLayout::SinglePE();
    std::vector<int64_t> tensor_shape = {100, 200};

    auto tile = layout.tile_shape({0, 0}, tensor_shape);
    EXPECT_EQ(tile.size(), 2);
    EXPECT_EQ(tile[0], 100);
    EXPECT_EQ(tile[1], 200);
}

TEST(MeshLayoutTest, TileShapeRowPartition) {
    auto layout = MeshLayout::RowPartition(4);
    std::vector<int64_t> tensor_shape = {100, 200};

    // First PE
    auto tile0 = layout.tile_shape({0, 0}, tensor_shape);
    EXPECT_EQ(tile0[0], 25);  // 100 / 4
    EXPECT_EQ(tile0[1], 200);

    // Last PE
    auto tile3 = layout.tile_shape({3, 0}, tensor_shape);
    EXPECT_EQ(tile3[0], 25);
    EXPECT_EQ(tile3[1], 200);
}

TEST(MeshLayoutTest, TileShapeGrid) {
    auto layout = MeshLayout::Grid(4, 4);
    std::vector<int64_t> tensor_shape = {100, 200};

    // PE(0, 0)
    auto tile00 = layout.tile_shape({0, 0}, tensor_shape);
    EXPECT_EQ(tile00[0], 25);   // 100 / 4
    EXPECT_EQ(tile00[1], 50);   // 200 / 4

    // PE(3, 3)
    auto tile33 = layout.tile_shape({3, 3}, tensor_shape);
    EXPECT_EQ(tile33[0], 25);
    EXPECT_EQ(tile33[1], 50);
}

TEST(MeshLayoutTest, MemoryPerPE) {
    auto layout = MeshLayout::Grid(4, 4);
    std::vector<int64_t> tensor_shape = {100, 200};
    size_t elem_size = 4;  // float32

    size_t mem = layout.memory_per_pe(tensor_shape, elem_size);
    // Each tile is 25 * 50 = 1250 elements = 5000 bytes
    EXPECT_EQ(mem, 5000);
}

TEST(MeshLayoutTest, Compatibility) {
    auto layout1 = MeshLayout::Grid(4, 4);
    auto layout2 = MeshLayout::Grid(4, 4);
    auto layout3 = MeshLayout::Grid(4, 8);
    auto layout4 = MeshLayout::RowPartition(4);

    EXPECT_TRUE(layout1.compatible_with(layout2));
    EXPECT_FALSE(layout1.compatible_with(layout3));
    EXPECT_FALSE(layout1.compatible_with(layout4));
}

TEST(MeshLayoutTest, ToString) {
    EXPECT_EQ(MeshLayout::SinglePE().to_string(), "SinglePE");
    EXPECT_EQ(MeshLayout::RowPartition(4).to_string(), "RowPartition(4)");
    EXPECT_EQ(MeshLayout::ColPartition(8).to_string(), "ColPartition(8)");
    EXPECT_EQ(MeshLayout::Grid(4, 8).to_string(), "Grid(4, 8)");
}
