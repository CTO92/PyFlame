#include "pyflame/utils/serialize.hpp"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <limits>

namespace pyflame::utils {

// ============================================================================
// Security: Safe Arithmetic Helpers
// ============================================================================

namespace {

/// Maximum allowed tensor size in bytes (16 GB) to prevent memory exhaustion
constexpr uint64_t MAX_TENSOR_BYTES = 16ULL * 1024 * 1024 * 1024;

/// Maximum allowed number of dimensions
constexpr size_t MAX_NDIM = 32;

/// Maximum allowed dimension size
constexpr int64_t MAX_DIM_SIZE = 1LL << 40;  // ~1 trillion

/// Safely multiply two int64_t values with overflow detection
/// Returns false if overflow would occur, true otherwise
inline bool safe_multiply_i64(int64_t a, int64_t b, int64_t& result) {
    if (a == 0 || b == 0) {
        result = 0;
        return true;
    }
    // Check for overflow before multiplication
    if (a > 0) {
        if (b > 0) {
            if (a > std::numeric_limits<int64_t>::max() / b) return false;
        } else {
            if (b < std::numeric_limits<int64_t>::min() / a) return false;
        }
    } else {
        if (b > 0) {
            if (a < std::numeric_limits<int64_t>::min() / b) return false;
        } else {
            if (a != 0 && b < std::numeric_limits<int64_t>::max() / a) return false;
        }
    }
    result = a * b;
    return true;
}

/// Safely multiply uint64_t values with overflow detection
inline bool safe_multiply_u64(uint64_t a, uint64_t b, uint64_t& result) {
    if (a == 0 || b == 0) {
        result = 0;
        return true;
    }
    if (a > std::numeric_limits<uint64_t>::max() / b) return false;
    result = a * b;
    return true;
}

/// Calculate number of elements from shape with overflow checking
/// Throws std::runtime_error on overflow or invalid dimensions
int64_t safe_numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 1;  // Scalar
    }

    if (shape.size() > MAX_NDIM) {
        throw std::runtime_error(
            "Tensor has too many dimensions (" + std::to_string(shape.size()) +
            "), maximum allowed: " + std::to_string(MAX_NDIM)
        );
    }

    int64_t numel = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("Negative dimension size: " + std::to_string(dim));
        }
        if (dim > MAX_DIM_SIZE) {
            throw std::runtime_error(
                "Dimension size too large: " + std::to_string(dim) +
                ", maximum: " + std::to_string(MAX_DIM_SIZE)
            );
        }
        if (!safe_multiply_i64(numel, dim, numel)) {
            throw std::runtime_error("Integer overflow calculating tensor size");
        }
    }
    return numel;
}

/// Calculate tensor size in bytes with overflow checking
uint64_t safe_tensor_bytes(int64_t numel, size_t dtype_bytes) {
    if (numel < 0) {
        throw std::runtime_error("Negative number of elements");
    }

    uint64_t total_bytes;
    if (!safe_multiply_u64(static_cast<uint64_t>(numel), dtype_bytes, total_bytes)) {
        throw std::runtime_error("Integer overflow calculating tensor byte size");
    }

    if (total_bytes > MAX_TENSOR_BYTES) {
        throw std::runtime_error(
            "Tensor size (" + std::to_string(total_bytes / (1024*1024)) +
            " MB) exceeds maximum allowed (" +
            std::to_string(MAX_TENSOR_BYTES / (1024*1024*1024)) + " GB)"
        );
    }

    return total_bytes;
}

}  // anonymous namespace

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

    // Security: Limit number of tensors to prevent memory exhaustion
    constexpr uint64_t MAX_TENSORS = 100000;
    if (header.num_tensors > MAX_TENSORS) {
        throw std::runtime_error("PyFlame file contains too many tensors (" +
            std::to_string(header.num_tensors) + "), maximum: " +
            std::to_string(MAX_TENSORS));
    }

    // Read tensor index
    std::vector<TensorEntry> entries;
    entries.reserve(header.num_tensors);

    for (uint64_t i = 0; i < header.num_tensors; ++i) {
        TensorEntry entry;

        // Read name with length validation
        uint32_t name_len;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

        // Security: Limit name length to prevent memory exhaustion
        constexpr uint32_t MAX_NAME_LEN = 4096;
        if (name_len > MAX_NAME_LEN) {
            throw std::runtime_error("Tensor name too long: " + std::to_string(name_len));
        }
        entry.name.resize(name_len);
        in.read(entry.name.data(), name_len);

        // Read dtype
        uint8_t dtype_val;
        in.read(reinterpret_cast<char*>(&dtype_val), sizeof(dtype_val));
        entry.dtype = static_cast<DType>(dtype_val);

        // Read shape with validation
        uint8_t ndim;
        in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

        // Security: Validate number of dimensions
        if (ndim > MAX_NDIM) {
            throw std::runtime_error("Tensor '" + entry.name +
                "' has too many dimensions: " + std::to_string(ndim));
        }

        entry.shape.resize(ndim);
        for (uint8_t j = 0; j < ndim; ++j) {
            in.read(reinterpret_cast<char*>(&entry.shape[j]), sizeof(int64_t));

            // Security: Validate each dimension
            if (entry.shape[j] < 0 || entry.shape[j] > MAX_DIM_SIZE) {
                throw std::runtime_error("Tensor '" + entry.name +
                    "' has invalid dimension size: " + std::to_string(entry.shape[j]));
            }
        }

        // Read offset and size
        in.read(reinterpret_cast<char*>(&entry.data_offset), sizeof(entry.data_offset));
        in.read(reinterpret_cast<char*>(&entry.data_size), sizeof(entry.data_size));

        // Security: Validate data size using safe arithmetic
        int64_t expected_numel = safe_numel(entry.shape);
        uint64_t expected_bytes = safe_tensor_bytes(expected_numel, dtype_size(entry.dtype));
        if (entry.data_size != expected_bytes) {
            throw std::runtime_error("Tensor '" + entry.name +
                "' data size mismatch (expected " + std::to_string(expected_bytes) +
                ", got " + std::to_string(entry.data_size) + ")");
        }

        entries.push_back(std::move(entry));
    }

    // Read tensor data
    StateDict result;
    for (const auto& entry : entries) {
        // Allocate buffer and read data (size already validated)
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

/// Simple JSON value type for SafeTensors header parsing
struct JsonValue {
    enum Type { NONE, STRING, NUMBER, ARRAY, OBJECT };
    Type type = NONE;
    std::string str_val;
    int64_t num_val = 0;
    std::vector<JsonValue> arr_val;
    std::map<std::string, JsonValue> obj_val;
};

/// Minimal JSON parser for SafeTensors headers (security-hardened)
class SafeTensorsJsonParser {
public:
    explicit SafeTensorsJsonParser(const std::string& json) : json_(json), pos_(0) {}

    JsonValue parse() {
        skip_whitespace();
        return parse_value();
    }

private:
    const std::string& json_;
    size_t pos_;

    // Maximum allowed nesting depth to prevent stack overflow attacks
    static constexpr int MAX_DEPTH = 10;
    int depth_ = 0;

    char peek() const { return pos_ < json_.size() ? json_[pos_] : '\0'; }
    char get() { return pos_ < json_.size() ? json_[pos_++] : '\0'; }

    void skip_whitespace() {
        while (pos_ < json_.size() && std::isspace(json_[pos_])) pos_++;
    }

    JsonValue parse_value() {
        if (++depth_ > MAX_DEPTH) {
            throw std::runtime_error("SafeTensors: invalid file format (structure too deep)");
        }

        skip_whitespace();
        JsonValue val;
        char c = peek();

        if (c == '"') {
            val.type = JsonValue::STRING;
            val.str_val = parse_string();
        } else if (c == '{') {
            val.type = JsonValue::OBJECT;
            val.obj_val = parse_object();
        } else if (c == '[') {
            val.type = JsonValue::ARRAY;
            val.arr_val = parse_array();
        } else if (c == '-' || std::isdigit(c)) {
            val.type = JsonValue::NUMBER;
            val.num_val = parse_number();
        } else {
            // Security: Don't leak position information in error messages
            throw std::runtime_error("SafeTensors: invalid file format (malformed header)");
        }

        depth_--;
        return val;
    }

    std::string parse_string() {
        if (get() != '"') throw std::runtime_error("SafeTensors: invalid file format");
        std::string result;
        // Limit string length to prevent memory exhaustion
        static constexpr size_t MAX_STRING_LEN = 1024 * 1024;  // 1MB
        while (pos_ < json_.size() && result.size() < MAX_STRING_LEN) {
            char c = get();
            if (c == '"') return result;
            if (c == '\\') {
                c = get();
                switch (c) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'r': result += '\r'; break;
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    default: result += c;
                }
            } else {
                result += c;
            }
        }
        throw std::runtime_error("SafeTensors: invalid file format (string too long or unterminated)");
    }

    int64_t parse_number() {
        std::string num_str;
        if (peek() == '-') num_str += get();
        while (std::isdigit(peek())) num_str += get();
        // Skip decimal part if present (we only need integers for SafeTensors)
        if (peek() == '.') {
            get();
            while (std::isdigit(peek())) get();
        }
        return std::stoll(num_str);
    }

    std::map<std::string, JsonValue> parse_object() {
        if (get() != '{') throw std::runtime_error("SafeTensors: invalid file format");
        std::map<std::string, JsonValue> obj;
        // Limit object size to prevent memory exhaustion
        static constexpr size_t MAX_ENTRIES = 10000;

        skip_whitespace();
        if (peek() == '}') { get(); return obj; }

        while (obj.size() < MAX_ENTRIES) {
            skip_whitespace();
            std::string key = parse_string();
            skip_whitespace();
            if (get() != ':') throw std::runtime_error("SafeTensors: invalid file format");
            obj[key] = parse_value();
            skip_whitespace();
            char c = get();
            if (c == '}') return obj;
            if (c != ',') throw std::runtime_error("SafeTensors: invalid file format");
        }
        throw std::runtime_error("SafeTensors: invalid file format (too many entries)");
    }

    std::vector<JsonValue> parse_array() {
        if (get() != '[') throw std::runtime_error("SafeTensors: invalid file format");
        std::vector<JsonValue> arr;
        // Limit array size to prevent memory exhaustion
        static constexpr size_t MAX_ELEMENTS = 100000;

        skip_whitespace();
        if (peek() == ']') { get(); return arr; }

        while (arr.size() < MAX_ELEMENTS) {
            arr.push_back(parse_value());
            skip_whitespace();
            char c = get();
            if (c == ']') return arr;
            if (c != ',') throw std::runtime_error("SafeTensors: invalid file format");
        }
        throw std::runtime_error("SafeTensors: invalid file format (too many elements)");
    }
};

/// Convert SafeTensors dtype string to PyFlame DType
DType safetensors_dtype_to_dtype(const std::string& dtype_str) {
    // SafeTensors uses uppercase dtype names
    if (dtype_str == "F32" || dtype_str == "float32") return DType::Float32;
    if (dtype_str == "F16" || dtype_str == "float16") return DType::Float16;
    if (dtype_str == "BF16" || dtype_str == "bfloat16") return DType::BFloat16;
    if (dtype_str == "I32" || dtype_str == "int32") return DType::Int32;
    if (dtype_str == "I16" || dtype_str == "int16") return DType::Int16;
    if (dtype_str == "I8" || dtype_str == "int8") return DType::Int8;
    if (dtype_str == "U8" || dtype_str == "uint8") return DType::UInt8;
    if (dtype_str == "I64" || dtype_str == "int64") return DType::Int64;
    if (dtype_str == "F64" || dtype_str == "float64") return DType::Float64;
    if (dtype_str == "BOOL" || dtype_str == "bool") return DType::Bool;
    throw std::runtime_error("SafeTensors: unsupported dtype: " + dtype_str);
}

StateDict read_safetensors(std::istream& in) {
    // Validate stream is readable
    if (!in.good()) {
        throw std::runtime_error("SafeTensors: invalid input stream");
    }

    // Read header size (8 bytes, little endian)
    uint64_t header_size;
    in.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!in.good()) {
        throw std::runtime_error("SafeTensors: failed to read header size");
    }

    // Security: Limit header size to prevent memory exhaustion (max 100MB)
    static constexpr uint64_t MAX_HEADER_SIZE = 100 * 1024 * 1024;
    if (header_size > MAX_HEADER_SIZE) {
        throw std::runtime_error("SafeTensors: header size exceeds maximum allowed (" +
                                 std::to_string(header_size) + " > " +
                                 std::to_string(MAX_HEADER_SIZE) + ")");
    }

    // Read header JSON
    std::string header_str(header_size, '\0');
    in.read(header_str.data(), header_size);
    if (!in.good()) {
        throw std::runtime_error("SafeTensors: failed to read header JSON");
    }

    // Parse header JSON
    SafeTensorsJsonParser parser(header_str);
    JsonValue header;
    try {
        header = parser.parse();
    } catch (const std::exception& e) {
        throw std::runtime_error("SafeTensors: failed to parse header: " +
                                 std::string(e.what()));
    }

    if (header.type != JsonValue::OBJECT) {
        throw std::runtime_error("SafeTensors: header must be a JSON object");
    }

    // Calculate data start position
    std::streampos data_start = static_cast<std::streampos>(8 + header_size);

    // Parse tensor entries and read data
    StateDict result;
    for (const auto& [name, value] : header.obj_val) {
        // Skip metadata entry
        if (name == "__metadata__") continue;

        if (value.type != JsonValue::OBJECT) {
            throw std::runtime_error("SafeTensors: tensor entry must be an object");
        }

        const auto& tensor_info = value.obj_val;

        // Get dtype
        auto dtype_it = tensor_info.find("dtype");
        if (dtype_it == tensor_info.end() || dtype_it->second.type != JsonValue::STRING) {
            throw std::runtime_error("SafeTensors: tensor '" + name + "' missing dtype");
        }
        DType dtype = safetensors_dtype_to_dtype(dtype_it->second.str_val);

        // Get shape
        auto shape_it = tensor_info.find("shape");
        if (shape_it == tensor_info.end() || shape_it->second.type != JsonValue::ARRAY) {
            throw std::runtime_error("SafeTensors: tensor '" + name + "' missing shape");
        }
        std::vector<int64_t> shape;
        for (const auto& dim : shape_it->second.arr_val) {
            if (dim.type != JsonValue::NUMBER) {
                throw std::runtime_error("SafeTensors: invalid shape dimension");
            }
            // Security: Validate dimension value before adding
            if (dim.num_val < 0 || dim.num_val > MAX_DIM_SIZE) {
                throw std::runtime_error("SafeTensors: invalid dimension size for tensor '" + name + "'");
            }
            shape.push_back(dim.num_val);
        }

        // Security: Validate number of dimensions
        if (shape.size() > MAX_NDIM) {
            throw std::runtime_error("SafeTensors: tensor '" + name +
                                     "' has too many dimensions (" +
                                     std::to_string(shape.size()) + ")");
        }

        // Get data offsets [start, end]
        auto offsets_it = tensor_info.find("data_offsets");
        if (offsets_it == tensor_info.end() || offsets_it->second.type != JsonValue::ARRAY ||
            offsets_it->second.arr_val.size() != 2) {
            throw std::runtime_error("SafeTensors: tensor '" + name +
                                     "' missing or invalid data_offsets");
        }
        uint64_t data_offset_start = static_cast<uint64_t>(offsets_it->second.arr_val[0].num_val);
        uint64_t data_offset_end = static_cast<uint64_t>(offsets_it->second.arr_val[1].num_val);

        // Security: Validate offset ordering
        if (data_offset_end < data_offset_start) {
            throw std::runtime_error("SafeTensors: tensor '" + name +
                                     "' has invalid data offsets (end < start)");
        }
        uint64_t data_size = data_offset_end - data_offset_start;

        // Security: Validate data size matches shape using safe arithmetic
        int64_t expected_numel = safe_numel(shape);
        uint64_t expected_bytes = safe_tensor_bytes(expected_numel, dtype_size(dtype));
        if (data_size != expected_bytes) {
            throw std::runtime_error("SafeTensors: tensor '" + name +
                                     "' data size mismatch (expected " +
                                     std::to_string(expected_bytes) + ", got " +
                                     std::to_string(data_size) + ")");
        }

        // Read tensor data
        std::vector<uint8_t> buffer(data_size);
        in.seekg(data_start + static_cast<std::streamoff>(data_offset_start));
        if (!in.good()) {
            throw std::runtime_error("SafeTensors: failed to seek to tensor data");
        }
        in.read(reinterpret_cast<char*>(buffer.data()), data_size);
        if (!in.good()) {
            throw std::runtime_error("SafeTensors: failed to read tensor data");
        }

        // Create tensor from data
        Tensor tensor = Tensor::from_data(
            buffer.data(),
            shape,
            dtype,
            MeshLayout::SinglePE(),
            data_size
        );

        result[name] = std::move(tensor);
    }

    return result;
}

/// Get NumPy dtype string from PyFlame DType
std::string dtype_to_numpy_descr(DType dtype) {
    // NumPy dtype descriptors: '<' = little-endian, '>' = big-endian
    // Using little-endian as it's most common
    switch (dtype) {
        case DType::Float32: return "<f4";
        case DType::Float64: return "<f8";
        case DType::Int8: return "<i1";
        case DType::Int16: return "<i2";
        case DType::Int32: return "<i4";
        case DType::Int64: return "<i8";
        case DType::UInt8: return "<u1";
        case DType::Bool: return "|b1";
        case DType::Float16: return "<f2";
        case DType::BFloat16: return "<V2";  // Custom 2-byte type
        default:
            throw std::runtime_error("Unsupported dtype for NumPy format: " + dtype_name(dtype));
    }
}

/// Parse NumPy dtype string to PyFlame DType
DType numpy_descr_to_dtype(const std::string& descr) {
    // Handle both with and without endianness prefix
    std::string d = descr;
    if (d.size() > 1 && (d[0] == '<' || d[0] == '>' || d[0] == '|' || d[0] == '=')) {
        d = d.substr(1);
    }

    if (d == "f4" || d == "float32") return DType::Float32;
    if (d == "f8" || d == "float64") return DType::Float64;
    if (d == "i1" || d == "int8") return DType::Int8;
    if (d == "i2" || d == "int16") return DType::Int16;
    if (d == "i4" || d == "int32") return DType::Int32;
    if (d == "i8" || d == "int64") return DType::Int64;
    if (d == "u1" || d == "uint8") return DType::UInt8;
    if (d == "b1" || d == "bool") return DType::Bool;
    if (d == "f2" || d == "float16") return DType::Float16;

    throw std::runtime_error("Unsupported NumPy dtype: " + descr);
}

/// Write a single tensor in NPY format
void write_npy(const Tensor& tensor, std::ostream& out) {
    // NPY format v1.0:
    // - Magic: \x93NUMPY
    // - Version: 1.0 (2 bytes)
    // - Header length: 2 bytes (little-endian)
    // - Header: Python dict literal
    // - Padding to 64-byte alignment
    // - Raw data

    // Build header dictionary
    std::ostringstream header;
    header << "{'descr': '" << dtype_to_numpy_descr(tensor.dtype()) << "', ";
    header << "'fortran_order': False, ";
    header << "'shape': (";

    auto shape = tensor.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        header << shape[i];
        if (i < shape.size() - 1) header << ", ";
        else if (shape.size() == 1) header << ",";  // Single-element tuple needs trailing comma
    }
    header << "), }";

    std::string header_str = header.str();

    // Calculate padding for 64-byte alignment
    // Total header = 6 (magic) + 2 (version) + 2 (header len) + header + padding + newline
    size_t base_len = 6 + 2 + 2 + header_str.size() + 1;  // +1 for newline
    size_t padding = (64 - (base_len % 64)) % 64;
    header_str += std::string(padding, ' ');
    header_str += '\n';

    // Write magic number
    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);

    // Write version (1.0)
    uint8_t version_major = 1;
    uint8_t version_minor = 0;
    out.write(reinterpret_cast<const char*>(&version_major), 1);
    out.write(reinterpret_cast<const char*>(&version_minor), 1);

    // Write header length (little-endian, 2 bytes for v1.0)
    uint16_t header_len = static_cast<uint16_t>(header_str.size());
    out.write(reinterpret_cast<const char*>(&header_len), 2);

    // Write header
    out.write(header_str.data(), header_str.size());

    // Write tensor data
    Tensor t = tensor;
    t.eval();
    const void* data = t.data_ptr();
    size_t data_bytes = static_cast<size_t>(t.numel()) * dtype_size(t.dtype());
    out.write(static_cast<const char*>(data), data_bytes);
}

/// Read a single tensor from NPY format
Tensor read_npy(std::istream& in) {
    // Read and verify magic number
    char magic[6];
    in.read(magic, 6);
    if (!in.good() || magic[0] != '\x93' || std::strncmp(magic + 1, "NUMPY", 5) != 0) {
        throw std::runtime_error("Invalid NPY file: bad magic number");
    }

    // Read version
    uint8_t version_major, version_minor;
    in.read(reinterpret_cast<char*>(&version_major), 1);
    in.read(reinterpret_cast<char*>(&version_minor), 1);

    // Read header length
    uint32_t header_len;
    if (version_major == 1) {
        uint16_t len16;
        in.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else if (version_major == 2) {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("Unsupported NPY version: " +
                                  std::to_string(version_major) + "." +
                                  std::to_string(version_minor));
    }

    // Security: Limit header size
    if (header_len > 1024 * 1024) {
        throw std::runtime_error("NPY header too large");
    }

    // Read header
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);

    // Parse header (simple parser for the expected format)
    // Find 'descr': '...'
    DType dtype = DType::Float32;
    auto descr_pos = header.find("'descr':");
    if (descr_pos != std::string::npos) {
        auto quote1 = header.find('\'', descr_pos + 8);
        auto quote2 = header.find('\'', quote1 + 1);
        if (quote1 != std::string::npos && quote2 != std::string::npos) {
            std::string descr = header.substr(quote1 + 1, quote2 - quote1 - 1);
            dtype = numpy_descr_to_dtype(descr);
        }
    }

    // Find 'shape': (...)
    std::vector<int64_t> shape;
    auto shape_pos = header.find("'shape':");
    if (shape_pos != std::string::npos) {
        auto paren1 = header.find('(', shape_pos);
        auto paren2 = header.find(')', paren1);
        if (paren1 != std::string::npos && paren2 != std::string::npos) {
            std::string shape_str = header.substr(paren1 + 1, paren2 - paren1 - 1);
            // Parse comma-separated integers
            std::istringstream ss(shape_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                // Trim whitespace
                size_t start = token.find_first_not_of(" \t");
                size_t end = token.find_last_not_of(" \t");
                if (start != std::string::npos && end != std::string::npos) {
                    token = token.substr(start, end - start + 1);
                    if (!token.empty()) {
                        shape.push_back(std::stoll(token));
                    }
                }
            }
        }
    }

    // Validate shape
    int64_t numel = safe_numel(shape);
    size_t data_bytes = safe_tensor_bytes(numel, dtype_size(dtype));

    // Read tensor data
    std::vector<uint8_t> buffer(data_bytes);
    in.read(reinterpret_cast<char*>(buffer.data()), data_bytes);
    if (!in.good()) {
        throw std::runtime_error("Failed to read NPY tensor data");
    }

    return Tensor::from_data(buffer.data(), shape, dtype, MeshLayout::SinglePE(), data_bytes);
}

/// Simple ZIP file writer for NPZ format
/// NPZ is just a ZIP archive containing .npy files
class SimpleZipWriter {
public:
    explicit SimpleZipWriter(std::ostream& out) : out_(out), offset_(0) {}

    void add_file(const std::string& name, const std::vector<uint8_t>& data) {
        FileEntry entry;
        entry.name = name;
        entry.offset = offset_;
        entry.size = data.size();

        // Local file header
        write32(0x04034b50);  // Signature
        write16(20);          // Version needed
        write16(0);           // Flags
        write16(0);           // Compression (none)
        write16(0);           // Mod time
        write16(0);           // Mod date
        write32(crc32(data)); // CRC-32
        write32(static_cast<uint32_t>(data.size())); // Compressed size
        write32(static_cast<uint32_t>(data.size())); // Uncompressed size
        write16(static_cast<uint16_t>(name.size())); // File name length
        write16(0);           // Extra field length

        out_.write(name.data(), name.size());
        out_.write(reinterpret_cast<const char*>(data.data()), data.size());

        offset_ += 30 + name.size() + data.size();
        entries_.push_back(entry);
    }

    void finish() {
        uint32_t cd_offset = offset_;

        // Write central directory
        for (const auto& entry : entries_) {
            write32(0x02014b50);  // Signature
            write16(20);          // Version made by
            write16(20);          // Version needed
            write16(0);           // Flags
            write16(0);           // Compression
            write16(0);           // Mod time
            write16(0);           // Mod date
            write32(0);           // CRC (placeholder)
            write32(static_cast<uint32_t>(entry.size)); // Compressed
            write32(static_cast<uint32_t>(entry.size)); // Uncompressed
            write16(static_cast<uint16_t>(entry.name.size())); // Name length
            write16(0);           // Extra length
            write16(0);           // Comment length
            write16(0);           // Disk number
            write16(0);           // Internal attrs
            write32(0);           // External attrs
            write32(static_cast<uint32_t>(entry.offset)); // Offset

            out_.write(entry.name.data(), entry.name.size());
            offset_ += 46 + entry.name.size();
        }

        uint32_t cd_size = offset_ - cd_offset;

        // End of central directory
        write32(0x06054b50);  // Signature
        write16(0);           // Disk number
        write16(0);           // CD disk
        write16(static_cast<uint16_t>(entries_.size())); // Entries on disk
        write16(static_cast<uint16_t>(entries_.size())); // Total entries
        write32(cd_size);     // CD size
        write32(cd_offset);   // CD offset
        write16(0);           // Comment length
    }

private:
    struct FileEntry {
        std::string name;
        uint32_t offset;
        size_t size;
    };

    void write16(uint16_t v) {
        out_.write(reinterpret_cast<const char*>(&v), 2);
    }

    void write32(uint32_t v) {
        out_.write(reinterpret_cast<const char*>(&v), 4);
    }

    uint32_t crc32(const std::vector<uint8_t>& data) {
        // Simple CRC32 implementation
        uint32_t crc = 0xFFFFFFFF;
        for (uint8_t byte : data) {
            crc ^= byte;
            for (int i = 0; i < 8; ++i) {
                crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
            }
        }
        return ~crc;
    }

    std::ostream& out_;
    uint32_t offset_;
    std::vector<FileEntry> entries_;
};

void write_numpy(const StateDict& state_dict, const std::string& path) {
    // Check if single tensor (.npy) or multiple (.npz)
    bool is_npz = path.size() >= 4 && path.substr(path.size() - 4) == ".npz";

    if (!is_npz && state_dict.size() == 1) {
        // Single tensor - write as .npy
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open file for writing: " + path);
        }
        write_npy(state_dict.begin()->second, out);
        return;
    }

    // Multiple tensors - write as .npz (ZIP of .npy files)
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    SimpleZipWriter zip(out);

    for (const auto& [name, tensor] : state_dict) {
        // Write tensor to memory buffer
        std::ostringstream npy_stream(std::ios::binary);
        write_npy(tensor, npy_stream);
        std::string npy_data = npy_stream.str();

        std::vector<uint8_t> data(npy_data.begin(), npy_data.end());
        zip.add_file(name + ".npy", data);
    }

    zip.finish();
}

/// Simple ZIP reader for NPZ format
StateDict read_numpy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    // Check if it's a plain .npy file
    char magic[6];
    in.read(magic, 6);
    in.seekg(0);

    if (magic[0] == '\x93' && std::strncmp(magic + 1, "NUMPY", 5) == 0) {
        // Single .npy file
        StateDict result;
        result["arr_0"] = read_npy(in);
        return result;
    }

    // It's a ZIP file (.npz)
    StateDict result;

    // Read local file headers
    while (in.good()) {
        uint32_t sig;
        in.read(reinterpret_cast<char*>(&sig), 4);

        if (sig != 0x04034b50) {
            // Not a local file header, might be central directory
            break;
        }

        // Skip version, flags, compression, time, date
        in.seekg(10, std::ios::cur);

        uint32_t crc, comp_size, uncomp_size;
        in.read(reinterpret_cast<char*>(&crc), 4);
        in.read(reinterpret_cast<char*>(&comp_size), 4);
        in.read(reinterpret_cast<char*>(&uncomp_size), 4);

        uint16_t name_len, extra_len;
        in.read(reinterpret_cast<char*>(&name_len), 2);
        in.read(reinterpret_cast<char*>(&extra_len), 2);

        // Security: Validate sizes
        if (name_len > 4096 || uncomp_size > MAX_TENSOR_BYTES) {
            throw std::runtime_error("NPZ: invalid entry size");
        }

        // Read filename
        std::string filename(name_len, '\0');
        in.read(filename.data(), name_len);

        // Skip extra field
        in.seekg(extra_len, std::ios::cur);

        // Read file data
        std::vector<uint8_t> data(uncomp_size);
        in.read(reinterpret_cast<char*>(data.data()), uncomp_size);

        // Parse the .npy data
        if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".npy") {
            std::string tensor_name = filename.substr(0, filename.size() - 4);
            std::istringstream npy_stream(std::string(data.begin(), data.end()));
            result[tensor_name] = read_npy(npy_stream);
        }
    }

    return result;
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
