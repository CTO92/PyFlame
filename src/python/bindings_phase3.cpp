/**
 * Python bindings for Phase 3: Model Support
 *
 * Provides Python access to:
 * - Data loading (Dataset, DataLoader)
 * - Model serialization (save/load)
 * - Pre-built models (ResNet, Transformer, BERT)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "pyflame/data/dataset.hpp"
#include "pyflame/data/dataloader.hpp"
#include "pyflame/utils/serialize.hpp"
#include "pyflame/models/resnet.hpp"
#include "pyflame/models/transformer.hpp"

namespace py = pybind11;

using namespace pyflame;
using namespace pyflame::data;
using namespace pyflame::utils;
using namespace pyflame::models;

// =============================================================================
// Data Loading Bindings
// =============================================================================

void bind_data_module(py::module_& m) {
    auto data = m.def_submodule("data", "Data loading utilities");

    // Sample type
    py::class_<Sample>(data, "Sample")
        .def(py::init<>())
        .def_readwrite("data", &Sample::data)
        .def_readwrite("target", &Sample::target)
        .def_readwrite("metadata", &Sample::metadata);

    // Batch type
    py::class_<Batch>(data, "Batch")
        .def(py::init<>())
        .def_readwrite("data", &Batch::data)
        .def_readwrite("targets", &Batch::targets)
        .def("size", &Batch::size);

    // DataLoaderOptions
    py::class_<DataLoaderOptions>(data, "DataLoaderOptions")
        .def(py::init<>())
        .def_readwrite("batch_size", &DataLoaderOptions::batch_size)
        .def_readwrite("shuffle", &DataLoaderOptions::shuffle)
        .def_readwrite("drop_last", &DataLoaderOptions::drop_last)
        .def_readwrite("num_workers", &DataLoaderOptions::num_workers)
        .def_readwrite("seed", &DataLoaderOptions::seed);

    // Dataset base class (for inheritance)
    py::class_<Dataset, std::shared_ptr<Dataset>>(data, "Dataset")
        .def("get_item", &Dataset::get_item)
        .def("size", &Dataset::size)
        .def("__len__", &Dataset::size)
        .def("__getitem__", &Dataset::get_item);

    // TensorDataset
    py::class_<TensorDataset, Dataset, std::shared_ptr<TensorDataset>>(
        data, "TensorDataset"
    )
        .def(py::init<const Tensor&>())
        .def(py::init<const Tensor&, const Tensor&>())
        .def("get_item", &TensorDataset::get_item)
        .def("size", &TensorDataset::size);

    // DataLoader
    py::class_<DataLoader>(data, "DataLoader")
        .def(py::init<std::shared_ptr<Dataset>, DataLoaderOptions>(),
             py::arg("dataset"),
             py::arg("options") = DataLoaderOptions())
        .def("__iter__", [](DataLoader& self) {
            return py::make_iterator(self.begin(), self.end());
        }, py::keep_alive<0, 1>())
        .def("__len__", &DataLoader::num_batches)
        .def_property_readonly("batch_size", &DataLoader::batch_size);
}

// =============================================================================
// Serialization Bindings
// =============================================================================

void bind_serialize_module(py::module_& m) {
    auto utils = m.def_submodule("utils", "Utility functions");

    // SerializeFormat enum
    py::enum_<SerializeFormat>(utils, "SerializeFormat")
        .value("PYFLAME_NATIVE", SerializeFormat::PYFLAME_NATIVE)
        .value("SAFETENSORS", SerializeFormat::SAFETENSORS)
        .value("NUMPY", SerializeFormat::NUMPY);

    // SerializeOptions
    py::class_<SerializeOptions>(utils, "SerializeOptions")
        .def(py::init<>())
        .def_readwrite("format", &SerializeOptions::format)
        .def_readwrite("compress", &SerializeOptions::compress)
        .def_readwrite("include_metadata", &SerializeOptions::include_metadata);

    // Checkpoint
    py::class_<Checkpoint>(utils, "Checkpoint")
        .def(py::init<>())
        .def_readwrite("model_state", &Checkpoint::model_state)
        .def_readwrite("optimizer_state", &Checkpoint::optimizer_state)
        .def_readwrite("epoch", &Checkpoint::epoch)
        .def_readwrite("global_step", &Checkpoint::global_step)
        .def_readwrite("best_metric", &Checkpoint::best_metric)
        .def_readwrite("metadata", &Checkpoint::metadata)
        .def("save", &Checkpoint::save)
        .def_static("load", &Checkpoint::load);

    // Save/Load functions
    utils.def("save", py::overload_cast<const StateDict&, const std::string&, SerializeOptions>(
        &save),
        py::arg("state_dict"),
        py::arg("path"),
        py::arg("options") = SerializeOptions(),
        "Save a state dict to file");

    utils.def("save_module", py::overload_cast<const nn::Module&, const std::string&, SerializeOptions>(
        &save),
        py::arg("module"),
        py::arg("path"),
        py::arg("options") = SerializeOptions(),
        "Save a module's state dict to file");

    utils.def("load", &load,
        py::arg("path"),
        py::arg("options") = SerializeOptions(),
        "Load a state dict from file");

    utils.def("load_into", &load_into,
        py::arg("module"),
        py::arg("path"),
        py::arg("strict") = true,
        "Load weights into an existing module");

    utils.def("save_checkpoint", &save_checkpoint,
        py::arg("checkpoint"),
        py::arg("path"),
        "Save a training checkpoint");

    utils.def("load_checkpoint", &load_checkpoint,
        py::arg("path"),
        "Load a training checkpoint");

    utils.def("detect_format", &detect_format,
        py::arg("path"),
        "Detect file format from extension");
}

// =============================================================================
// ResNet Model Bindings
// =============================================================================

void bind_resnet_models(py::module_& m) {
    auto models = m.def_submodule("models", "Pre-built model architectures");

    // ResNetConfig
    py::class_<ResNetConfig>(models, "ResNetConfig")
        .def(py::init<>())
        .def_readwrite("name", &ResNetConfig::name)
        .def_readwrite("layers", &ResNetConfig::layers)
        .def_readwrite("use_bottleneck", &ResNetConfig::use_bottleneck)
        .def_readwrite("num_classes", &ResNetConfig::num_classes)
        .def_readwrite("groups", &ResNetConfig::groups)
        .def_readwrite("width_per_group", &ResNetConfig::width_per_group)
        .def_readwrite("zero_init_residual", &ResNetConfig::zero_init_residual)
        .def_static("ResNet18", &ResNetConfig::ResNet18,
            py::arg("num_classes") = 1000)
        .def_static("ResNet34", &ResNetConfig::ResNet34,
            py::arg("num_classes") = 1000)
        .def_static("ResNet50", &ResNetConfig::ResNet50,
            py::arg("num_classes") = 1000)
        .def_static("ResNet101", &ResNetConfig::ResNet101,
            py::arg("num_classes") = 1000)
        .def_static("ResNet152", &ResNetConfig::ResNet152,
            py::arg("num_classes") = 1000)
        .def_static("ResNeXt50_32x4d", &ResNetConfig::ResNeXt50_32x4d,
            py::arg("num_classes") = 1000)
        .def_static("ResNeXt101_32x8d", &ResNetConfig::ResNeXt101_32x8d,
            py::arg("num_classes") = 1000)
        .def_static("WideResNet50_2", &ResNetConfig::WideResNet50_2,
            py::arg("num_classes") = 1000)
        .def_static("WideResNet101_2", &ResNetConfig::WideResNet101_2,
            py::arg("num_classes") = 1000);

    // ResNet model
    py::class_<ResNet, nn::Module, std::shared_ptr<ResNet>>(models, "ResNet")
        .def(py::init<const ResNetConfig&>())
        .def("forward", py::overload_cast<const Tensor&>(&ResNet::forward))
        .def("forward_features", &ResNet::forward_features)
        .def("num_features", &ResNet::num_features)
        .def("num_classes", &ResNet::num_classes)
        .def("reset_classifier", &ResNet::reset_classifier)
        .def("__call__", py::overload_cast<const Tensor&>(&ResNet::forward));

    // Factory functions
    models.def("resnet18", &resnet18,
        py::arg("num_classes") = 1000,
        "Create ResNet-18 model");

    models.def("resnet34", &resnet34,
        py::arg("num_classes") = 1000,
        "Create ResNet-34 model");

    models.def("resnet50", &resnet50,
        py::arg("num_classes") = 1000,
        "Create ResNet-50 model");

    models.def("resnet101", &resnet101,
        py::arg("num_classes") = 1000,
        "Create ResNet-101 model");

    models.def("resnet152", &resnet152,
        py::arg("num_classes") = 1000,
        "Create ResNet-152 model");

    models.def("resnext50_32x4d", &resnext50_32x4d,
        py::arg("num_classes") = 1000,
        "Create ResNeXt-50 32x4d model");

    models.def("resnext101_32x8d", &resnext101_32x8d,
        py::arg("num_classes") = 1000,
        "Create ResNeXt-101 32x8d model");

    models.def("wide_resnet50_2", &wide_resnet50_2,
        py::arg("num_classes") = 1000,
        "Create Wide ResNet-50-2 model");

    models.def("wide_resnet101_2", &wide_resnet101_2,
        py::arg("num_classes") = 1000,
        "Create Wide ResNet-101-2 model");
}

// =============================================================================
// Transformer Model Bindings
// =============================================================================

void bind_transformer_models(py::module_& m) {
    auto models = m.def_submodule("models");

    // TransformerConfig
    py::class_<TransformerConfig>(models, "TransformerConfig")
        .def(py::init<>())
        .def_readwrite("d_model", &TransformerConfig::d_model)
        .def_readwrite("nhead", &TransformerConfig::nhead)
        .def_readwrite("num_encoder_layers", &TransformerConfig::num_encoder_layers)
        .def_readwrite("num_decoder_layers", &TransformerConfig::num_decoder_layers)
        .def_readwrite("dim_feedforward", &TransformerConfig::dim_feedforward)
        .def_readwrite("dropout", &TransformerConfig::dropout)
        .def_readwrite("activation", &TransformerConfig::activation)
        .def_readwrite("batch_first", &TransformerConfig::batch_first)
        .def_readwrite("norm_first", &TransformerConfig::norm_first)
        .def_static("Base", &TransformerConfig::Base)
        .def_static("Large", &TransformerConfig::Large);

    // MultiHeadAttention
    py::class_<MultiHeadAttention, nn::Module, std::shared_ptr<MultiHeadAttention>>(
        models, "MultiHeadAttention"
    )
        .def(py::init<int64_t, int64_t, float, bool, bool, bool, int64_t, int64_t, bool>(),
            py::arg("embed_dim"),
            py::arg("num_heads"),
            py::arg("dropout") = 0.0f,
            py::arg("bias") = true,
            py::arg("add_bias_kv") = false,
            py::arg("add_zero_attn") = false,
            py::arg("kdim") = -1,
            py::arg("vdim") = -1,
            py::arg("batch_first") = true)
        .def("forward", py::overload_cast<const Tensor&>(&MultiHeadAttention::forward));

    // TransformerEncoderLayer
    py::class_<TransformerEncoderLayer, nn::Module, std::shared_ptr<TransformerEncoderLayer>>(
        models, "TransformerEncoderLayer"
    )
        .def(py::init<int64_t, int64_t, int64_t, float, const std::string&, float, bool, bool>(),
            py::arg("d_model"),
            py::arg("nhead"),
            py::arg("dim_feedforward") = 2048,
            py::arg("dropout") = 0.1f,
            py::arg("activation") = "relu",
            py::arg("layer_norm_eps") = 1e-5f,
            py::arg("batch_first") = true,
            py::arg("norm_first") = false)
        .def("forward", py::overload_cast<const Tensor&>(&TransformerEncoderLayer::forward));

    // TransformerDecoderLayer
    py::class_<TransformerDecoderLayer, nn::Module, std::shared_ptr<TransformerDecoderLayer>>(
        models, "TransformerDecoderLayer"
    )
        .def(py::init<int64_t, int64_t, int64_t, float, const std::string&, float, bool, bool>(),
            py::arg("d_model"),
            py::arg("nhead"),
            py::arg("dim_feedforward") = 2048,
            py::arg("dropout") = 0.1f,
            py::arg("activation") = "relu",
            py::arg("layer_norm_eps") = 1e-5f,
            py::arg("batch_first") = true,
            py::arg("norm_first") = false);

    // Transformer
    py::class_<Transformer, nn::Module, std::shared_ptr<Transformer>>(
        models, "Transformer"
    )
        .def(py::init<const TransformerConfig&>(),
            py::arg("config") = TransformerConfig())
        .def_static("generate_square_subsequent_mask",
            &Transformer::generate_square_subsequent_mask,
            py::arg("sz"),
            "Generate causal mask for autoregressive decoding");

    // BertConfig
    py::class_<BertConfig>(models, "BertConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &BertConfig::vocab_size)
        .def_readwrite("hidden_size", &BertConfig::hidden_size)
        .def_readwrite("num_hidden_layers", &BertConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &BertConfig::num_attention_heads)
        .def_readwrite("intermediate_size", &BertConfig::intermediate_size)
        .def_readwrite("hidden_dropout_prob", &BertConfig::hidden_dropout_prob)
        .def_readwrite("attention_probs_dropout_prob", &BertConfig::attention_probs_dropout_prob)
        .def_readwrite("max_position_embeddings", &BertConfig::max_position_embeddings)
        .def_readwrite("type_vocab_size", &BertConfig::type_vocab_size)
        .def_readwrite("layer_norm_eps", &BertConfig::layer_norm_eps)
        .def_readwrite("pad_token_id", &BertConfig::pad_token_id)
        .def_static("Base", &BertConfig::Base)
        .def_static("Large", &BertConfig::Large);

    // BertModel
    py::class_<BertModel, nn::Module, std::shared_ptr<BertModel>>(
        models, "BertModel"
    )
        .def(py::init<const BertConfig&>())
        .def("forward", py::overload_cast<const Tensor&>(&BertModel::forward))
        .def("hidden_size", &BertModel::hidden_size);

    // BertForSequenceClassification
    py::class_<BertForSequenceClassification, nn::Module, std::shared_ptr<BertForSequenceClassification>>(
        models, "BertForSequenceClassification"
    )
        .def(py::init<const BertConfig&, int64_t>(),
            py::arg("config"),
            py::arg("num_labels"))
        .def("forward", py::overload_cast<const Tensor&>(&BertForSequenceClassification::forward));

    // BertForMaskedLM
    py::class_<BertForMaskedLM, nn::Module, std::shared_ptr<BertForMaskedLM>>(
        models, "BertForMaskedLM"
    )
        .def(py::init<const BertConfig&>())
        .def("forward", py::overload_cast<const Tensor&>(&BertForMaskedLM::forward));

    // PositionalEncoding
    py::class_<PositionalEncoding, nn::Module, std::shared_ptr<PositionalEncoding>>(
        models, "PositionalEncoding"
    )
        .def(py::init<int64_t, float, int64_t>(),
            py::arg("d_model"),
            py::arg("dropout") = 0.1f,
            py::arg("max_len") = 5000)
        .def("forward", py::overload_cast<const Tensor&>(&PositionalEncoding::forward));

    // Factory functions
    models.def("bert_base", &bert_base,
        "Create BERT base model");

    models.def("bert_large", &bert_large,
        "Create BERT large model");

    models.def("transformer_base", &transformer_base,
        "Create base Transformer model");

    models.def("transformer_large", &transformer_large,
        "Create large Transformer model");
}

// =============================================================================
// Module Registration
// =============================================================================

PYBIND11_MODULE(_pyflame_phase3, m) {
    m.doc() = "PyFlame Phase 3: Model Support";

    // Bind all submodules
    bind_data_module(m);
    bind_serialize_module(m);
    bind_resnet_models(m);
    bind_transformer_models(m);
}
