#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <stdexcept>

#include "pyflame/core/tensor.hpp"

namespace pyflame::data {

/// A single sample from a dataset (data, label pair)
struct Sample {
    Tensor data;
    Tensor label;

    Sample() = default;
    Sample(Tensor d, Tensor l) : data(std::move(d)), label(std::move(l)) {}
};

/// Abstract base class for all datasets
class Dataset {
public:
    virtual ~Dataset() = default;

    /// Get a single item by index
    virtual Sample get_item(int64_t index) const = 0;

    /// Total number of samples in the dataset
    virtual int64_t size() const = 0;

    /// Alias for size() - Python __len__ compatibility
    int64_t __len__() const { return size(); }

    /// Get item - Python __getitem__ compatibility
    Sample __getitem__(int64_t index) const { return get_item(index); }
};

/// Dataset that wraps in-memory tensors
class TensorDataset : public Dataset {
public:
    /// Create from data and labels tensors
    /// @param data Tensor of shape [N, ...] containing all data samples
    /// @param labels Tensor of shape [N, ...] containing all labels
    TensorDataset(Tensor data, Tensor labels)
        : data_(std::move(data)), labels_(std::move(labels)) {
        if (data_.shape()[0] != labels_.shape()[0]) {
            throw std::invalid_argument(
                "Data and labels must have same first dimension. Got " +
                std::to_string(data_.shape()[0]) + " vs " +
                std::to_string(labels_.shape()[0]));
        }
    }

    /// Create from multiple tensors (all must have same first dimension)
    explicit TensorDataset(std::vector<Tensor> tensors) : tensors_(std::move(tensors)) {
        if (tensors_.empty()) {
            throw std::invalid_argument("TensorDataset requires at least one tensor");
        }
        int64_t n = tensors_[0].shape()[0];
        for (size_t i = 1; i < tensors_.size(); ++i) {
            if (tensors_[i].shape()[0] != n) {
                throw std::invalid_argument(
                    "All tensors must have same first dimension");
            }
        }
    }

    Sample get_item(int64_t index) const override {
        if (index < 0 || index >= size()) {
            throw std::out_of_range("Dataset index out of range: " +
                std::to_string(index) + " >= " + std::to_string(size()));
        }

        if (!tensors_.empty()) {
            // Multi-tensor mode: return first two as data/label
            return Sample(
                tensors_[0][index],
                tensors_.size() > 1 ? tensors_[1][index] : Tensor()
            );
        }
        // Dual tensor mode
        return Sample(data_[index], labels_[index]);
    }

    int64_t size() const override {
        if (!tensors_.empty()) {
            return tensors_[0].shape()[0];
        }
        return data_.shape()[0];
    }

    /// Access underlying tensors
    const Tensor& data() const { return data_; }
    const Tensor& labels() const { return labels_; }
    const std::vector<Tensor>& tensors() const { return tensors_; }

private:
    Tensor data_;
    Tensor labels_;
    std::vector<Tensor> tensors_;
};

/// Dataset that loads samples lazily using a callback
class LazyDataset : public Dataset {
public:
    using LoadFn = std::function<Sample(int64_t)>;

    /// Create lazy dataset
    /// @param size Total number of samples
    /// @param load_fn Function to load a sample by index
    LazyDataset(int64_t size, LoadFn load_fn)
        : size_(size), load_fn_(std::move(load_fn)) {
        if (size_ < 0) {
            throw std::invalid_argument("Dataset size cannot be negative");
        }
    }

    Sample get_item(int64_t index) const override {
        if (index < 0 || index >= size_) {
            throw std::out_of_range("Dataset index out of range");
        }
        return load_fn_(index);
    }

    int64_t size() const override { return size_; }

private:
    int64_t size_;
    LoadFn load_fn_;
};

/// Subset of another dataset
class Subset : public Dataset {
public:
    /// Create subset from dataset and indices
    Subset(std::shared_ptr<Dataset> dataset, std::vector<int64_t> indices)
        : dataset_(std::move(dataset)), indices_(std::move(indices)) {
        // Validate indices
        for (auto idx : indices_) {
            if (idx < 0 || idx >= dataset_->size()) {
                throw std::out_of_range(
                    "Subset index out of range: " + std::to_string(idx));
            }
        }
    }

    Sample get_item(int64_t index) const override {
        if (index < 0 || index >= static_cast<int64_t>(indices_.size())) {
            throw std::out_of_range("Subset index out of range");
        }
        return dataset_->get_item(indices_[index]);
    }

    int64_t size() const override {
        return static_cast<int64_t>(indices_.size());
    }

    /// Get the underlying dataset
    std::shared_ptr<Dataset> dataset() const { return dataset_; }

    /// Get the indices
    const std::vector<int64_t>& indices() const { return indices_; }

private:
    std::shared_ptr<Dataset> dataset_;
    std::vector<int64_t> indices_;
};

/// Concatenation of multiple datasets
class ConcatDataset : public Dataset {
public:
    explicit ConcatDataset(std::vector<std::shared_ptr<Dataset>> datasets)
        : datasets_(std::move(datasets)) {
        if (datasets_.empty()) {
            throw std::invalid_argument("ConcatDataset requires at least one dataset");
        }

        // Compute cumulative sizes for efficient indexing
        cumulative_sizes_.reserve(datasets_.size());
        int64_t total = 0;
        for (const auto& ds : datasets_) {
            total += ds->size();
            cumulative_sizes_.push_back(total);
        }
    }

    Sample get_item(int64_t index) const override {
        if (index < 0 || index >= size()) {
            throw std::out_of_range("ConcatDataset index out of range");
        }

        // Binary search to find which dataset contains this index
        size_t ds_idx = 0;
        while (ds_idx < cumulative_sizes_.size() && index >= cumulative_sizes_[ds_idx]) {
            ++ds_idx;
        }

        int64_t local_idx = (ds_idx == 0) ? index : index - cumulative_sizes_[ds_idx - 1];
        return datasets_[ds_idx]->get_item(local_idx);
    }

    int64_t size() const override {
        return cumulative_sizes_.empty() ? 0 : cumulative_sizes_.back();
    }

    /// Get the underlying datasets
    const std::vector<std::shared_ptr<Dataset>>& datasets() const { return datasets_; }

private:
    std::vector<std::shared_ptr<Dataset>> datasets_;
    std::vector<int64_t> cumulative_sizes_;
};

/// Split a dataset into train/validation subsets
inline std::pair<std::shared_ptr<Subset>, std::shared_ptr<Subset>> random_split(
    std::shared_ptr<Dataset> dataset,
    double train_ratio,
    uint64_t seed = 0
) {
    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        throw std::invalid_argument("train_ratio must be between 0 and 1 (exclusive)");
    }

    int64_t n = dataset->size();
    int64_t train_size = static_cast<int64_t>(n * train_ratio);

    // Generate shuffled indices
    std::vector<int64_t> indices(n);
    for (int64_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Split indices
    std::vector<int64_t> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<int64_t> val_indices(indices.begin() + train_size, indices.end());

    return {
        std::make_shared<Subset>(dataset, std::move(train_indices)),
        std::make_shared<Subset>(dataset, std::move(val_indices))
    };
}

}  // namespace pyflame::data
