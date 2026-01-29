// PyFlame Phase 2: Neural Network Python bindings

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "pyflame/core/tensor.hpp"
#include "pyflame/autograd/grad_mode.hpp"
#include "pyflame/autograd/autograd.hpp"
#include "pyflame/nn/module.hpp"
#include "pyflame/nn/linear.hpp"
#include "pyflame/nn/conv.hpp"
#include "pyflame/nn/normalization.hpp"
#include "pyflame/nn/pooling.hpp"
#include "pyflame/nn/dropout.hpp"
#include "pyflame/nn/attention.hpp"
#include "pyflame/nn/loss.hpp"
#include "pyflame/optim/optimizer.hpp"
#include "pyflame/optim/lr_scheduler.hpp"

namespace py = pybind11;
using namespace pyflame;
using namespace pyflame::nn;
using namespace pyflame::optim;
using namespace pyflame::autograd;

void init_nn_bindings(py::module& m) {
    // ========================================================================
    // Autograd
    // ========================================================================
    auto autograd_m = m.def_submodule("autograd", "Automatic differentiation");

    py::class_<GradMode>(autograd_m, "GradMode")
        .def_static("is_enabled", &GradMode::is_enabled,
            "Check if gradient computation is enabled")
        .def_static("set_enabled", &GradMode::set_enabled,
            py::arg("enabled"), "Set gradient computation mode");

    py::class_<NoGradGuard>(autograd_m, "no_grad")
        .def(py::init<>())
        .def("__enter__", [](NoGradGuard& self) { return &self; })
        .def("__exit__", [](NoGradGuard&, py::object, py::object, py::object) { });

    autograd_m.def("backward", [](Tensor& output, const Tensor& grad_output) {
        AutogradEngine::backward(output.node(), grad_output.node(), output.graph());
    }, py::arg("output"), py::arg("grad_output") = Tensor(),
    "Compute gradients for the given output tensor");

    // ========================================================================
    // nn submodule
    // ========================================================================
    auto nn_m = m.def_submodule("nn", "Neural network modules");

    // Reduction enum for losses
    py::enum_<Reduction>(nn_m, "Reduction")
        .value("NONE", Reduction::NONE)
        .value("MEAN", Reduction::MEAN)
        .value("SUM", Reduction::SUM)
        .export_values();

    // -------------------------------------------------------------------------
    // Module base class
    // -------------------------------------------------------------------------
    py::class_<Module, std::shared_ptr<Module>>(nn_m, "Module")
        .def("forward", static_cast<Tensor(Module::*)(const Tensor&)>(&Module::forward),
            py::arg("input"), "Forward pass")
        .def("__call__", static_cast<Tensor(Module::*)(const Tensor&)>(&Module::forward),
            py::arg("input"))
        .def("parameters", &Module::parameters, "Get all learnable parameters")
        .def("zero_grad", &Module::zero_grad, "Zero out all parameter gradients")
        .def("train", &Module::train, py::arg("mode") = true, "Set training mode")
        .def("eval", &Module::eval, "Set evaluation mode")
        .def("is_training", &Module::is_training, "Check if in training mode")
        .def("name", &Module::name, "Get module name")
        .def("state_dict", &Module::state_dict, "Get state dictionary")
        .def("load_state_dict", &Module::load_state_dict,
            py::arg("dict"), py::arg("strict") = true, "Load state dictionary")
        .def("__repr__", &Module::to_string);

    // -------------------------------------------------------------------------
    // Sequential container
    // -------------------------------------------------------------------------
    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(nn_m, "Sequential")
        .def(py::init<>())
        .def(py::init([](std::vector<std::shared_ptr<Module>> modules) {
            auto seq = std::make_shared<Sequential>();
            for (auto& m : modules) {
                seq->add(m);
            }
            return seq;
        }), py::arg("modules"))
        .def("add", &Sequential::add, py::arg("module"), "Add a module")
        .def("__len__", &Sequential::size)
        .def("__getitem__", [](const Sequential& self, size_t index) {
            return self[index];
        }, py::arg("index"));

    // -------------------------------------------------------------------------
    // Linear layer
    // -------------------------------------------------------------------------
    py::class_<Linear, Module, std::shared_ptr<Linear>>(nn_m, "Linear")
        .def(py::init<int64_t, int64_t, bool>(),
            py::arg("in_features"),
            py::arg("out_features"),
            py::arg("bias") = true,
            "Linear transformation: y = x @ W^T + b")
        .def_property_readonly("in_features", &Linear::in_features)
        .def_property_readonly("out_features", &Linear::out_features)
        .def_property_readonly("weight", static_cast<const Tensor& (Linear::*)() const>(&Linear::weight))
        .def_property_readonly("bias", static_cast<const Tensor& (Linear::*)() const>(&Linear::bias));

    // -------------------------------------------------------------------------
    // Convolution layers
    // -------------------------------------------------------------------------
    py::class_<Conv2d, Module, std::shared_ptr<Conv2d>>(nn_m, "Conv2d")
        .def(py::init<int64_t, int64_t, std::array<int64_t, 2>,
                      std::array<int64_t, 2>, std::array<int64_t, 2>,
                      std::array<int64_t, 2>, int64_t, bool>(),
            py::arg("in_channels"),
            py::arg("out_channels"),
            py::arg("kernel_size"),
            py::arg("stride") = std::array<int64_t, 2>{1, 1},
            py::arg("padding") = std::array<int64_t, 2>{0, 0},
            py::arg("dilation") = std::array<int64_t, 2>{1, 1},
            py::arg("groups") = 1,
            py::arg("bias") = true)
        .def_property_readonly("in_channels", &Conv2d::in_channels)
        .def_property_readonly("out_channels", &Conv2d::out_channels)
        .def_property_readonly("kernel_size", &Conv2d::kernel_size)
        .def_property_readonly("weight", static_cast<const Tensor& (Conv2d::*)() const>(&Conv2d::weight));

    py::class_<Conv1d, Module, std::shared_ptr<Conv1d>>(nn_m, "Conv1d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool>(),
            py::arg("in_channels"),
            py::arg("out_channels"),
            py::arg("kernel_size"),
            py::arg("stride") = 1,
            py::arg("padding") = 0,
            py::arg("dilation") = 1,
            py::arg("groups") = 1,
            py::arg("bias") = true);

    // -------------------------------------------------------------------------
    // Normalization layers
    // -------------------------------------------------------------------------
    py::class_<BatchNorm2d, Module, std::shared_ptr<BatchNorm2d>>(nn_m, "BatchNorm2d")
        .def(py::init<int64_t, float, float, bool, bool>(),
            py::arg("num_features"),
            py::arg("eps") = 1e-5f,
            py::arg("momentum") = 0.1f,
            py::arg("affine") = true,
            py::arg("track_running_stats") = true)
        .def("reset_running_stats", &BatchNorm2d::reset_running_stats)
        .def_property_readonly("num_features", &BatchNorm2d::num_features);

    py::class_<BatchNorm1d, Module, std::shared_ptr<BatchNorm1d>>(nn_m, "BatchNorm1d")
        .def(py::init<int64_t, float, float, bool, bool>(),
            py::arg("num_features"),
            py::arg("eps") = 1e-5f,
            py::arg("momentum") = 0.1f,
            py::arg("affine") = true,
            py::arg("track_running_stats") = true)
        .def("reset_running_stats", &BatchNorm1d::reset_running_stats);

    py::class_<LayerNorm, Module, std::shared_ptr<LayerNorm>>(nn_m, "LayerNorm")
        .def(py::init<std::vector<int64_t>, float, bool>(),
            py::arg("normalized_shape"),
            py::arg("eps") = 1e-5f,
            py::arg("elementwise_affine") = true);

    py::class_<GroupNorm, Module, std::shared_ptr<GroupNorm>>(nn_m, "GroupNorm")
        .def(py::init<int64_t, int64_t, float, bool>(),
            py::arg("num_groups"),
            py::arg("num_channels"),
            py::arg("eps") = 1e-5f,
            py::arg("affine") = true);

    // -------------------------------------------------------------------------
    // Pooling layers
    // -------------------------------------------------------------------------
    py::class_<MaxPool2d, Module, std::shared_ptr<MaxPool2d>>(nn_m, "MaxPool2d")
        .def(py::init<std::array<int64_t, 2>, std::array<int64_t, 2>,
                      std::array<int64_t, 2>>(),
            py::arg("kernel_size"),
            py::arg("stride") = std::array<int64_t, 2>{0, 0},
            py::arg("padding") = std::array<int64_t, 2>{0, 0})
        .def(py::init<int64_t, int64_t, int64_t>(),
            py::arg("kernel_size"),
            py::arg("stride") = 0,
            py::arg("padding") = 0);

    py::class_<AvgPool2d, Module, std::shared_ptr<AvgPool2d>>(nn_m, "AvgPool2d")
        .def(py::init<std::array<int64_t, 2>, std::array<int64_t, 2>,
                      std::array<int64_t, 2>>(),
            py::arg("kernel_size"),
            py::arg("stride") = std::array<int64_t, 2>{0, 0},
            py::arg("padding") = std::array<int64_t, 2>{0, 0})
        .def(py::init<int64_t, int64_t, int64_t>(),
            py::arg("kernel_size"),
            py::arg("stride") = 0,
            py::arg("padding") = 0);

    py::class_<AdaptiveAvgPool2d, Module, std::shared_ptr<AdaptiveAvgPool2d>>(nn_m, "AdaptiveAvgPool2d")
        .def(py::init<std::array<int64_t, 2>>(),
            py::arg("output_size"));

    py::class_<GlobalAvgPool2d, Module, std::shared_ptr<GlobalAvgPool2d>>(nn_m, "GlobalAvgPool2d")
        .def(py::init<>());

    // -------------------------------------------------------------------------
    // Dropout layers
    // -------------------------------------------------------------------------
    py::class_<Dropout, Module, std::shared_ptr<Dropout>>(nn_m, "Dropout")
        .def(py::init<float, bool>(),
            py::arg("p") = 0.5f,
            py::arg("inplace") = false);

    py::class_<Dropout2d, Module, std::shared_ptr<Dropout2d>>(nn_m, "Dropout2d")
        .def(py::init<float, bool>(),
            py::arg("p") = 0.5f,
            py::arg("inplace") = false);

    // -------------------------------------------------------------------------
    // Attention layers
    // -------------------------------------------------------------------------
    py::class_<MultiheadAttention, Module, std::shared_ptr<MultiheadAttention>>(nn_m, "MultiheadAttention")
        .def(py::init<int64_t, int64_t, float, bool, bool, bool, int64_t, int64_t, bool>(),
            py::arg("embed_dim"),
            py::arg("num_heads"),
            py::arg("dropout") = 0.0f,
            py::arg("bias") = true,
            py::arg("add_bias_kv") = false,
            py::arg("add_zero_attn") = false,
            py::arg("kdim") = 0,
            py::arg("vdim") = 0,
            py::arg("batch_first") = false)
        .def("forward", static_cast<std::tuple<Tensor, Tensor>(MultiheadAttention::*)(
            const Tensor&, const Tensor&, const Tensor&, const Tensor&, bool)>(
            &MultiheadAttention::forward),
            py::arg("query"), py::arg("key"), py::arg("value"),
            py::arg("attn_mask") = Tensor(), py::arg("need_weights") = true);

    py::class_<SelfAttention, Module, std::shared_ptr<SelfAttention>>(nn_m, "SelfAttention")
        .def(py::init<int64_t, int64_t, float, bool>(),
            py::arg("embed_dim"),
            py::arg("num_heads"),
            py::arg("dropout") = 0.0f,
            py::arg("bias") = true);

    py::class_<CrossAttention, Module, std::shared_ptr<CrossAttention>>(nn_m, "CrossAttention")
        .def(py::init<int64_t, int64_t, float, bool>(),
            py::arg("embed_dim"),
            py::arg("num_heads"),
            py::arg("dropout") = 0.0f,
            py::arg("bias") = true)
        .def("forward", [](CrossAttention& self, const Tensor& query, const Tensor& context) {
            return self.forward(query, context);
        }, py::arg("query"), py::arg("context"));

    // -------------------------------------------------------------------------
    // Loss functions
    // -------------------------------------------------------------------------
    py::class_<MSELoss>(nn_m, "MSELoss")
        .def(py::init<Reduction>(), py::arg("reduction") = Reduction::MEAN)
        .def("forward", &MSELoss::forward, py::arg("input"), py::arg("target"))
        .def("__call__", &MSELoss::operator(), py::arg("input"), py::arg("target"))
        .def("__repr__", &MSELoss::to_string);

    py::class_<L1Loss>(nn_m, "L1Loss")
        .def(py::init<Reduction>(), py::arg("reduction") = Reduction::MEAN)
        .def("forward", &L1Loss::forward, py::arg("input"), py::arg("target"))
        .def("__call__", &L1Loss::operator(), py::arg("input"), py::arg("target"))
        .def("__repr__", &L1Loss::to_string);

    py::class_<SmoothL1Loss>(nn_m, "SmoothL1Loss")
        .def(py::init<Reduction, float>(),
            py::arg("reduction") = Reduction::MEAN,
            py::arg("beta") = 1.0f)
        .def("forward", &SmoothL1Loss::forward, py::arg("input"), py::arg("target"))
        .def("__call__", &SmoothL1Loss::operator(), py::arg("input"), py::arg("target"))
        .def("__repr__", &SmoothL1Loss::to_string);

    py::class_<BCELoss>(nn_m, "BCELoss")
        .def(py::init<Reduction>(), py::arg("reduction") = Reduction::MEAN)
        .def("forward", &BCELoss::forward, py::arg("input"), py::arg("target"))
        .def("__call__", &BCELoss::operator(), py::arg("input"), py::arg("target"))
        .def("__repr__", &BCELoss::to_string);

    py::class_<BCEWithLogitsLoss>(nn_m, "BCEWithLogitsLoss")
        .def(py::init<Reduction>(), py::arg("reduction") = Reduction::MEAN)
        .def("forward", &BCEWithLogitsLoss::forward, py::arg("input"), py::arg("target"))
        .def("__call__", &BCEWithLogitsLoss::operator(), py::arg("input"), py::arg("target"))
        .def("__repr__", &BCEWithLogitsLoss::to_string);

    py::class_<CrossEntropyLoss>(nn_m, "CrossEntropyLoss")
        .def(py::init<Reduction, int64_t, float>(),
            py::arg("reduction") = Reduction::MEAN,
            py::arg("ignore_index") = -100,
            py::arg("label_smoothing") = 0.0f)
        .def("forward", &CrossEntropyLoss::forward, py::arg("input"), py::arg("target"))
        .def("__call__", &CrossEntropyLoss::operator(), py::arg("input"), py::arg("target"))
        .def("__repr__", &CrossEntropyLoss::to_string);

    py::class_<NLLLoss>(nn_m, "NLLLoss")
        .def(py::init<Reduction, int64_t>(),
            py::arg("reduction") = Reduction::MEAN,
            py::arg("ignore_index") = -100)
        .def("forward", &NLLLoss::forward, py::arg("input"), py::arg("target"))
        .def("__call__", &NLLLoss::operator(), py::arg("input"), py::arg("target"))
        .def("__repr__", &NLLLoss::to_string);

    py::class_<KLDivLoss>(nn_m, "KLDivLoss")
        .def(py::init<Reduction, bool>(),
            py::arg("reduction") = Reduction::MEAN,
            py::arg("log_target") = false)
        .def("forward", &KLDivLoss::forward, py::arg("input"), py::arg("target"))
        .def("__call__", &KLDivLoss::operator(), py::arg("input"), py::arg("target"))
        .def("__repr__", &KLDivLoss::to_string);

    // Functional loss interface
    auto functional_m = nn_m.def_submodule("functional", "Functional interface");

    functional_m.def("mse_loss", &functional::mse_loss,
        py::arg("input"), py::arg("target"),
        py::arg("reduction") = Reduction::MEAN);
    functional_m.def("l1_loss", &functional::l1_loss,
        py::arg("input"), py::arg("target"),
        py::arg("reduction") = Reduction::MEAN);
    functional_m.def("cross_entropy", &functional::cross_entropy_loss,
        py::arg("input"), py::arg("target"),
        py::arg("reduction") = Reduction::MEAN,
        py::arg("ignore_index") = -100,
        py::arg("label_smoothing") = 0.0f);
    functional_m.def("bce_loss", &functional::bce_loss,
        py::arg("input"), py::arg("target"),
        py::arg("reduction") = Reduction::MEAN);
    functional_m.def("bce_with_logits", &functional::bce_with_logits_loss,
        py::arg("input"), py::arg("target"),
        py::arg("reduction") = Reduction::MEAN);

    // ========================================================================
    // optim submodule
    // ========================================================================
    auto optim_m = m.def_submodule("optim", "Optimizers and learning rate schedulers");

    // -------------------------------------------------------------------------
    // Optimizer base
    // -------------------------------------------------------------------------
    py::class_<Optimizer>(optim_m, "Optimizer")
        .def("step", &Optimizer::step, "Perform optimization step")
        .def("zero_grad", &Optimizer::zero_grad, "Zero all parameter gradients")
        .def("get_lr", &Optimizer::get_lr, "Get current learning rate")
        .def("set_lr", &Optimizer::set_lr, py::arg("lr"), "Set learning rate")
        .def("state_dict", &Optimizer::state_dict, "Get optimizer state")
        .def("load_state_dict", &Optimizer::load_state_dict, py::arg("state"))
        .def("__repr__", &Optimizer::to_string);

    // -------------------------------------------------------------------------
    // SGD
    // -------------------------------------------------------------------------
    py::class_<SGD, Optimizer>(optim_m, "SGD")
        .def(py::init<std::vector<Tensor*>, float, float, float, float, bool>(),
            py::arg("params"),
            py::arg("lr"),
            py::arg("momentum") = 0.0f,
            py::arg("dampening") = 0.0f,
            py::arg("weight_decay") = 0.0f,
            py::arg("nesterov") = false)
        .def_property_readonly("momentum", &SGD::momentum)
        .def_property_readonly("weight_decay", &SGD::weight_decay)
        .def_property_readonly("nesterov", &SGD::nesterov);

    // -------------------------------------------------------------------------
    // Adam
    // -------------------------------------------------------------------------
    py::class_<Adam, Optimizer>(optim_m, "Adam")
        .def(py::init<std::vector<Tensor*>, float, float, float, float, float, bool>(),
            py::arg("params"),
            py::arg("lr") = 0.001f,
            py::arg("beta1") = 0.9f,
            py::arg("beta2") = 0.999f,
            py::arg("eps") = 1e-8f,
            py::arg("weight_decay") = 0.0f,
            py::arg("amsgrad") = false)
        .def_property_readonly("beta1", &Adam::beta1)
        .def_property_readonly("beta2", &Adam::beta2)
        .def_property_readonly("eps", &Adam::eps)
        .def_property_readonly("weight_decay", &Adam::weight_decay)
        .def_property_readonly("amsgrad", &Adam::amsgrad);

    // -------------------------------------------------------------------------
    // AdamW
    // -------------------------------------------------------------------------
    py::class_<AdamW, Optimizer>(optim_m, "AdamW")
        .def(py::init<std::vector<Tensor*>, float, float, float, float, float, bool>(),
            py::arg("params"),
            py::arg("lr") = 0.001f,
            py::arg("beta1") = 0.9f,
            py::arg("beta2") = 0.999f,
            py::arg("eps") = 1e-8f,
            py::arg("weight_decay") = 0.01f,
            py::arg("amsgrad") = false)
        .def_property_readonly("beta1", &AdamW::beta1)
        .def_property_readonly("beta2", &AdamW::beta2)
        .def_property_readonly("weight_decay", &AdamW::weight_decay);

    // -------------------------------------------------------------------------
    // RMSprop
    // -------------------------------------------------------------------------
    py::class_<RMSprop, Optimizer>(optim_m, "RMSprop")
        .def(py::init<std::vector<Tensor*>, float, float, float, float, float, bool>(),
            py::arg("params"),
            py::arg("lr") = 0.01f,
            py::arg("alpha") = 0.99f,
            py::arg("eps") = 1e-8f,
            py::arg("weight_decay") = 0.0f,
            py::arg("momentum") = 0.0f,
            py::arg("centered") = false)
        .def_property_readonly("alpha", &RMSprop::alpha)
        .def_property_readonly("momentum", &RMSprop::momentum)
        .def_property_readonly("centered", &RMSprop::centered);

    // -------------------------------------------------------------------------
    // Learning rate schedulers
    // -------------------------------------------------------------------------
    py::class_<LRScheduler>(optim_m, "LRScheduler")
        .def("step", &LRScheduler::step, "Advance scheduler by one step")
        .def("get_lr", &LRScheduler::get_lr, "Get current learning rate")
        .def("last_epoch", &LRScheduler::last_epoch, "Get last epoch count")
        .def("base_lr", &LRScheduler::base_lr, "Get base learning rate")
        .def("__repr__", &LRScheduler::to_string);

    py::class_<StepLR, LRScheduler>(optim_m, "StepLR")
        .def(py::init<Optimizer&, int64_t, float, int64_t>(),
            py::arg("optimizer"),
            py::arg("step_size"),
            py::arg("gamma") = 0.1f,
            py::arg("last_epoch") = -1)
        .def_property_readonly("step_size", &StepLR::step_size)
        .def_property_readonly("gamma", &StepLR::gamma);

    py::class_<MultiStepLR, LRScheduler>(optim_m, "MultiStepLR")
        .def(py::init<Optimizer&, std::vector<int64_t>, float, int64_t>(),
            py::arg("optimizer"),
            py::arg("milestones"),
            py::arg("gamma") = 0.1f,
            py::arg("last_epoch") = -1)
        .def_property_readonly("milestones", &MultiStepLR::milestones)
        .def_property_readonly("gamma", &MultiStepLR::gamma);

    py::class_<ExponentialLR, LRScheduler>(optim_m, "ExponentialLR")
        .def(py::init<Optimizer&, float, int64_t>(),
            py::arg("optimizer"),
            py::arg("gamma"),
            py::arg("last_epoch") = -1)
        .def_property_readonly("gamma", &ExponentialLR::gamma);

    py::class_<CosineAnnealingLR, LRScheduler>(optim_m, "CosineAnnealingLR")
        .def(py::init<Optimizer&, int64_t, float, int64_t>(),
            py::arg("optimizer"),
            py::arg("T_max"),
            py::arg("eta_min") = 0.0f,
            py::arg("last_epoch") = -1)
        .def_property_readonly("T_max", &CosineAnnealingLR::T_max)
        .def_property_readonly("eta_min", &CosineAnnealingLR::eta_min);

    py::class_<ReduceLROnPlateau>(optim_m, "ReduceLROnPlateau")
        .def(py::init<Optimizer&, ReduceLROnPlateau::Mode, float, int64_t, float, int64_t, float>(),
            py::arg("optimizer"),
            py::arg("mode") = ReduceLROnPlateau::Mode::MIN,
            py::arg("factor") = 0.1f,
            py::arg("patience") = 10,
            py::arg("threshold") = 1e-4f,
            py::arg("cooldown") = 0,
            py::arg("min_lr") = 0.0f)
        .def("step", &ReduceLROnPlateau::step, py::arg("metric"))
        .def("get_lr", &ReduceLROnPlateau::get_lr)
        .def("__repr__", &ReduceLROnPlateau::to_string);

    py::enum_<ReduceLROnPlateau::Mode>(optim_m, "ReduceLROnPlateauMode")
        .value("MIN", ReduceLROnPlateau::Mode::MIN)
        .value("MAX", ReduceLROnPlateau::Mode::MAX)
        .export_values();

    py::class_<OneCycleLR, LRScheduler>(optim_m, "OneCycleLR")
        .def(py::init<Optimizer&, float, int64_t, float, OneCycleLR::AnnealStrategy, float, float, int64_t>(),
            py::arg("optimizer"),
            py::arg("max_lr"),
            py::arg("total_steps"),
            py::arg("pct_start") = 0.3f,
            py::arg("anneal_strategy") = OneCycleLR::AnnealStrategy::COS,
            py::arg("div_factor") = 25.0f,
            py::arg("final_div_factor") = 1e4f,
            py::arg("last_epoch") = -1);

    py::enum_<OneCycleLR::AnnealStrategy>(optim_m, "AnnealStrategy")
        .value("COS", OneCycleLR::AnnealStrategy::COS)
        .value("LINEAR", OneCycleLR::AnnealStrategy::LINEAR)
        .export_values();
}
