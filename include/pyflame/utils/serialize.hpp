#pragma once

#include <string>
#include <map>
#include <vector>
#include <cstdint>
#include <fstream>
#include <optional>

#include "pyflame/core/tensor.hpp"
#include "pyflame/nn/module.hpp"

namespace pyflame::utils {

// ============================================================================
// Serialization Format
// ============================================================================

/// Supported serialization formats
enum class SerializeFormat {
    PYFLAME_NATIVE,  // .pf - PyFlame native format
    SAFETENSORS,     // .safetensors - HuggingFace compatible
    NUMPY,           // .npz - NumPy compatible
};

/// Options for serialization
struct SerializeOptions {
    SerializeFormat format = SerializeFormat::PYFLAME_NATIVE;
    bool compress = false;           // Enable compression (zlib)
    int compression_level = 6;       // Compression level 1-9
    bool include_metadata = true;    // Include version info, etc.
};

// ============================================================================
// State Dictionary Types
// ============================================================================

/// Type alias for state dictionary
using StateDict = std::map<std::string, Tensor>;

// ============================================================================
// Save Functions
// ============================================================================

/// Save a state dictionary to file
/// @param state_dict Dictionary of name -> tensor mappings
/// @param path Output file path
/// @param options Serialization options
void save(
    const StateDict& state_dict,
    const std::string& path,
    SerializeOptions options = {}
);

/// Save a module's state dict to file
/// @param module Module to save
/// @param path Output file path
/// @param options Serialization options
void save(
    const nn::Module& module,
    const std::string& path,
    SerializeOptions options = {}
);

// ============================================================================
// Load Functions
// ============================================================================

/// Load a state dictionary from file
/// @param path Input file path
/// @param options Serialization options (format auto-detected if not specified)
/// @return Loaded state dictionary
StateDict load(
    const std::string& path,
    SerializeOptions options = {}
);

/// Load state dict into a module
/// @param module Module to load into
/// @param path Input file path
/// @param strict If true, throw on missing/unexpected keys
void load_into(
    nn::Module& module,
    const std::string& path,
    bool strict = true
);

// ============================================================================
// Checkpoint Management
// ============================================================================

/// Training checkpoint structure
struct Checkpoint {
    StateDict model_state;
    StateDict optimizer_state;
    int64_t epoch = 0;
    int64_t global_step = 0;
    double best_metric = std::numeric_limits<double>::infinity();
    std::map<std::string, std::string> metadata;

    /// Save checkpoint to file
    void save(const std::string& path) const;

    /// Load checkpoint from file
    static Checkpoint load(const std::string& path);
};

/// Save a checkpoint
void save_checkpoint(const Checkpoint& checkpoint, const std::string& path);

/// Load a checkpoint
Checkpoint load_checkpoint(const std::string& path);

// ============================================================================
// File Format Utilities
// ============================================================================

/// Detect format from file extension
SerializeFormat detect_format(const std::string& path);

/// Get file extension for format
std::string format_extension(SerializeFormat format);

// ============================================================================
// PyFlame Native Format Implementation
// ============================================================================

namespace detail {

/// Magic number for PyFlame native format
constexpr uint64_t PYFLAME_MAGIC = 0x50594641FFFF; // "PYFLAME\xFF\xFF"

/// File format version
constexpr uint32_t PYFLAME_VERSION = 1;

/// Header flags
enum HeaderFlags : uint32_t {
    FLAG_COMPRESSED = 1 << 0,
    FLAG_HAS_OPTIMIZER = 1 << 1,
    FLAG_HAS_METADATA = 1 << 2,
};

/// File header structure (64 bytes)
struct FileHeader {
    uint64_t magic;           // Magic number
    uint32_t version;         // Format version
    uint32_t flags;           // Header flags
    uint64_t num_tensors;     // Number of tensors
    uint64_t metadata_offset; // Offset to metadata section
    uint64_t data_offset;     // Offset to data section
    uint8_t reserved[24];     // Reserved for future use
};

/// Tensor entry in index
struct TensorEntry {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    uint64_t data_offset;
    uint64_t data_size;
};

/// Write tensor data in native format
void write_native(
    const StateDict& state_dict,
    std::ostream& out,
    const SerializeOptions& options
);

/// Read tensor data in native format
StateDict read_native(std::istream& in);

/// Write tensor data in safetensors format
void write_safetensors(
    const StateDict& state_dict,
    std::ostream& out
);

/// Read tensor data in safetensors format
StateDict read_safetensors(std::istream& in);

/// Write tensor data in numpy format (.npz)
void write_numpy(
    const StateDict& state_dict,
    const std::string& path
);

/// Read tensor data in numpy format (.npz)
StateDict read_numpy(const std::string& path);

}  // namespace detail

}  // namespace pyflame::utils
