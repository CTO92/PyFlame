#include "pyflame/utils/serialize.hpp"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <sstream>

namespace pyflame::utils {

// ============================================================================
// Format Detection
// ============================================================================

SerializeFormat detect_format(const std::string& path) {
    // Find the last dot for extension
    size_t dot_pos = path.rfind('.');
    if (dot_pos == std::string::npos) {
        return SerializeFormat::PYFLAME_NATIVE;
    }

    std::string ext = path.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".pf" || ext == ".pyflame") {
        return SerializeFormat::PYFLAME_NATIVE;
    } else if (ext == ".safetensors") {
        return SerializeFormat::SAFETENSORS;
    } else if (ext == ".npz" || ext == ".npy") {
        return SerializeFormat::NUMPY;
    }

    return SerializeFormat::PYFLAME_NATIVE;
}

std::string format_extension(SerializeFormat format) {
    switch (format) {
        case SerializeFormat::PYFLAME_NATIVE: return ".pf";
        case SerializeFormat::SAFETENSORS: return ".safetensors";
        case SerializeFormat::NUMPY: return ".npz";
        default: return ".pf";
    }
}

// ============================================================================
// Native Format Implementation
// ============================================================================

namespace detail {

void write_native(
    const StateDict& state_dict,
    std::ostream& out,
    const SerializeOptions& options
) {
    // Build tensor index
    std::vector<TensorEntry> entries;
    entries.reserve(state_dict.size());

    uint64_t current_offset = 0;
    for (const auto& [name, tensor] : state_dict) {
        TensorEntry entry;
        entry.name = name;
        entry.dtype = tensor.dtype();
        entry.shape = tensor.shape();

        // Calculate data size
        int64_t numel = tensor.numel();
        size_t dtype_bytes = dtype_size(tensor.dtype());
        entry.data_size = static_cast<uint64_t>(numel) * dtype_bytes;
        entry.data_offset = current_offset;

        current_offset += entry.data_size;
        entries.push_back(std::move(entry));
    }

    // Write header
    FileHeader header = {};
    header.magic = PYFLAME_MAGIC;
    header.version = PYFLAME_VERSION;
    header.flags = options.include_metadata ? FLAG_HAS_METADATA : 0;
    if (options.compress) {
        header.flags |= FLAG_COMPRESSED;
    }
    header.num_tensors = state_dict.size();

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write tensor index
    for (const auto& entry : entries) {
        // Write name (length-prefixed)
        uint32_t name_len = static_cast<uint32_t>(entry.name.size());
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        out.write(entry.name.data(), name_len);

        // Write dtype
        uint8_t dtype_val = static_cast<uint8_t>(entry.dtype);
        out.write(reinterpret_cast<const char*>(&dtype_val), sizeof(dtype_val));

        // Write shape (ndim + dims)
        uint8_t ndim = static_cast<uint8_t>(entry.shape.size());
        out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        for (int64_t dim : entry.shape) {
            out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        }

        // Write data offset and size
        out.write(reinterpret_cast<const char*>(&entry.data_offset), sizeof(entry.data_offset));
        out.write(reinterpret_cast<const char*>(&entry.data_size), sizeof(entry.data_size));
    }

    // Write tensor data
    for (const auto& [name, tensor] : state_dict) {
        // Force evaluation and get data pointer
        Tensor t = tensor;
        t.eval();
        const void* data = t.data_ptr();
        int64_t numel = t.numel();
        size_t dtype_bytes = dtype_size(t.dtype());
        size_t total_bytes = static_cast<size_t>(numel) * dtype_bytes;

        out.write(static_cast<const char*>(data), total_bytes);
    }

    // Write metadata as JSON if enabled
    if (options.include_metadata) {
        std::string metadata = R"({"pyflame_version":"0.1.0"})";
        uint64_t metadata_size = metadata.size();
        out.write(reinterpret_cast<const char*>(&metadata_size), sizeof(metadata_size));
        out.write(metadata.data(), metadata_size);
    }
}

StateDict read_native(std::istream& in) {
    // Read header
    FileHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (header.magic != PYFLAME_MAGIC) {
        throw std::runtime_error("Invalid PyFlame file: bad magic number");
    }

    if (header.version > PYFLAME_VERSION) {
        throw std::runtime_error("PyFlame file version " +
            std::to_string(header.version) + " is newer than supported version " +
            std::to_string(PYFLAME_VERSION));
    }

    // Read tensor index
    std::vector<TensorEntry> entries;
    entries.reserve(header.num_tensors);

    for (uint64_t i = 0; i < header.num_tensors; ++i) {
        TensorEntry entry;

        // Read name
        uint32_t name_len;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        entry.name.resize(name_len);
        in.read(entry.name.data(), name_len);

        // Read dtype
        uint8_t dtype_val;
        in.read(reinterpret_cast<char*>(&dtype_val), sizeof(dtype_val));
        entry.dtype = static_cast<DType>(dtype_val);

        // Read shape
        uint8_t ndim;
        in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        entry.shape.resize(ndim);
        for (uint8_t j = 0; j < ndim; ++j) {
            in.read(reinterpret_cast<char*>(&entry.shape[j]), sizeof(int64_t));
        }

        // Read offset and size
        in.read(reinterpret_cast<char*>(&entry.data_offset), sizeof(entry.data_offset));
        in.read(reinterpret_cast<char*>(&entry.data_size), sizeof(entry.data_size));

        entries.push_back(std::move(entry));
    }

    // Read tensor data
    StateDict result;
    for (const auto& entry : entries) {
        // Allocate buffer and read data
        std::vector<uint8_t> buffer(entry.data_size);
        in.read(reinterpret_cast<char*>(buffer.data()), entry.data_size);

        // Create tensor from data
        Tensor tensor = Tensor::from_data(
            buffer.data(),
            entry.shape,
            entry.dtype,
            MeshLayout::SinglePE(),
            entry.data_size
        );

        result[entry.name] = std::move(tensor);
    }

    return result;
}

void write_safetensors(const StateDict& state_dict, std::ostream& out) {
    // SafeTensors format:
    // 8 bytes: header size (little endian)
    // N bytes: JSON header with tensor metadata
    // Remaining: raw tensor data

    // Build header JSON
    std::ostringstream header_json;
    header_json << "{";

    uint64_t data_offset = 0;
    bool first = true;
    std::vector<std::pair<std::string, const Tensor*>> ordered_tensors;

    for (const auto& [name, tensor] : state_dict) {
        ordered_tensors.emplace_back(name, &tensor);
    }

    for (const auto& [name, tensor_ptr] : ordered_tensors) {
        const Tensor& tensor = *tensor_ptr;

        if (!first) header_json << ",";
        first = false;

        // Tensor entry
        header_json << "\"" << name << "\":{";
        header_json << "\"dtype\":\"" << dtype_name(tensor.dtype()) << "\",";
        header_json << "\"shape\":[";

        auto shape = tensor.shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) header_json << ",";
            header_json << shape[i];
        }
        header_json << "],";

        int64_t numel = tensor.numel();
        size_t data_size = static_cast<size_t>(numel) * dtype_size(tensor.dtype());

        header_json << "\"data_offsets\":[" << data_offset << "," << (data_offset + data_size) << "]";
        header_json << "}";

        data_offset += data_size;
    }

    // Add metadata
    header_json << ",\"__metadata__\":{\"format\":\"pf\",\"version\":\"0.1.0\"}";
    header_json << "}";

    std::string header_str = header_json.str();

    // Pad header to 8-byte alignment
    while (header_str.size() % 8 != 0) {
        header_str += ' ';
    }

    // Write header size
    uint64_t header_size = header_str.size();
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));

    // Write header
    out.write(header_str.data(), header_str.size());

    // Write tensor data
    for (const auto& [name, tensor_ptr] : ordered_tensors) {
        Tensor t = *tensor_ptr;
        t.eval();
        const void* data = t.data_ptr();
        int64_t numel = t.numel();
        size_t data_size = static_cast<size_t>(numel) * dtype_size(t.dtype());
        out.write(static_cast<const char*>(data), data_size);
    }
}

StateDict read_safetensors(std::istream& in) {
    // Read header size
    uint64_t header_size;
    in.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    // Read header JSON
    std::string header_str(header_size, '\0');
    in.read(header_str.data(), header_size);

    // Simple JSON parsing (production would use a proper JSON library)
    // This is a simplified implementation
    StateDict result;

    // For now, just return empty - proper implementation would parse JSON
    // and read tensor data based on offsets
    throw std::runtime_error("SafeTensors format reading not fully implemented yet");

    return result;
}

void write_numpy(const StateDict& state_dict, const std::string& path) {
    // NPZ format is a ZIP of .npy files
    // Simplified implementation - just save each tensor
    throw std::runtime_error("NumPy format not implemented yet");
}

StateDict read_numpy(const std::string& path) {
    throw std::runtime_error("NumPy format not implemented yet");
}

}  // namespace detail

// ============================================================================
// Public API Implementation
// ============================================================================

void save(const StateDict& state_dict, const std::string& path, SerializeOptions options) {
    // Auto-detect format from extension if not specified
    if (options.format == SerializeFormat::PYFLAME_NATIVE) {
        options.format = detect_format(path);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    switch (options.format) {
        case SerializeFormat::PYFLAME_NATIVE:
            detail::write_native(state_dict, out, options);
            break;
        case SerializeFormat::SAFETENSORS:
            detail::write_safetensors(state_dict, out);
            break;
        case SerializeFormat::NUMPY:
            out.close();
            detail::write_numpy(state_dict, path);
            break;
    }
}

void save(const nn::Module& module, const std::string& path, SerializeOptions options) {
    save(module.state_dict(), path, options);
}

StateDict load(const std::string& path, SerializeOptions options) {
    // Auto-detect format
    SerializeFormat format = detect_format(path);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    switch (format) {
        case SerializeFormat::PYFLAME_NATIVE:
            return detail::read_native(in);
        case SerializeFormat::SAFETENSORS:
            return detail::read_safetensors(in);
        case SerializeFormat::NUMPY:
            in.close();
            return detail::read_numpy(path);
    }

    throw std::runtime_error("Unknown format");
}

void load_into(nn::Module& module, const std::string& path, bool strict) {
    StateDict state_dict = load(path);
    module.load_state_dict(state_dict, strict);
}

// ============================================================================
// Checkpoint Implementation
// ============================================================================

void Checkpoint::save(const std::string& path) const {
    save_checkpoint(*this, path);
}

Checkpoint Checkpoint::load(const std::string& path) {
    return load_checkpoint(path);
}

void save_checkpoint(const Checkpoint& checkpoint, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open checkpoint file for writing: " + path);
    }

    // Write checkpoint header
    detail::FileHeader header = {};
    header.magic = detail::PYFLAME_MAGIC;
    header.version = detail::PYFLAME_VERSION;
    header.flags = detail::FLAG_HAS_OPTIMIZER | detail::FLAG_HAS_METADATA;
    header.num_tensors = checkpoint.model_state.size() + checkpoint.optimizer_state.size();

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write epoch and step
    out.write(reinterpret_cast<const char*>(&checkpoint.epoch), sizeof(checkpoint.epoch));
    out.write(reinterpret_cast<const char*>(&checkpoint.global_step), sizeof(checkpoint.global_step));
    out.write(reinterpret_cast<const char*>(&checkpoint.best_metric), sizeof(checkpoint.best_metric));

    // Write model state count
    uint64_t model_count = checkpoint.model_state.size();
    out.write(reinterpret_cast<const char*>(&model_count), sizeof(model_count));

    // Write optimizer state count
    uint64_t optim_count = checkpoint.optimizer_state.size();
    out.write(reinterpret_cast<const char*>(&optim_count), sizeof(optim_count));

    // Write model state
    SerializeOptions opts;
    opts.include_metadata = false;
    detail::write_native(checkpoint.model_state, out, opts);

    // Write optimizer state
    detail::write_native(checkpoint.optimizer_state, out, opts);

    // Write metadata
    std::ostringstream metadata_json;
    metadata_json << "{";
    bool first = true;
    for (const auto& [key, value] : checkpoint.metadata) {
        if (!first) metadata_json << ",";
        first = false;
        metadata_json << "\"" << key << "\":\"" << value << "\"";
    }
    metadata_json << "}";

    std::string metadata_str = metadata_json.str();
    uint64_t metadata_size = metadata_str.size();
    out.write(reinterpret_cast<const char*>(&metadata_size), sizeof(metadata_size));
    out.write(metadata_str.data(), metadata_size);
}

Checkpoint load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open checkpoint file for reading: " + path);
    }

    // Read header
    detail::FileHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (header.magic != detail::PYFLAME_MAGIC) {
        throw std::runtime_error("Invalid checkpoint file: bad magic number");
    }

    Checkpoint checkpoint;

    // Read epoch and step
    in.read(reinterpret_cast<char*>(&checkpoint.epoch), sizeof(checkpoint.epoch));
    in.read(reinterpret_cast<char*>(&checkpoint.global_step), sizeof(checkpoint.global_step));
    in.read(reinterpret_cast<char*>(&checkpoint.best_metric), sizeof(checkpoint.best_metric));

    // Read counts
    uint64_t model_count, optim_count;
    in.read(reinterpret_cast<char*>(&model_count), sizeof(model_count));
    in.read(reinterpret_cast<char*>(&optim_count), sizeof(optim_count));

    // Read model state
    checkpoint.model_state = detail::read_native(in);

    // Read optimizer state if present
    if (header.flags & detail::FLAG_HAS_OPTIMIZER) {
        checkpoint.optimizer_state = detail::read_native(in);
    }

    return checkpoint;
}

}  // namespace pyflame::utils
