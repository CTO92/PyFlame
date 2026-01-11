#include "pyflame/data/dataloader.hpp"

#include <algorithm>
#include <numeric>

namespace pyflame::data {

// ============================================================================
// Default collate function
// ============================================================================

Batch default_collate(const std::vector<Sample>& samples) {
    if (samples.empty()) {
        return Batch();
    }

    // Stack all data tensors along dim 0
    std::vector<Tensor> data_tensors;
    std::vector<Tensor> label_tensors;

    data_tensors.reserve(samples.size());
    label_tensors.reserve(samples.size());

    for (const auto& sample : samples) {
        data_tensors.push_back(sample.data);
        if (sample.label.numel() > 0) {
            label_tensors.push_back(sample.label);
        }
    }

    Tensor batched_data = stack(data_tensors, 0);
    Tensor batched_labels = label_tensors.empty() ? Tensor() : stack(label_tensors, 0);

    return Batch(std::move(batched_data), std::move(batched_labels));
}

// ============================================================================
// DataLoader Implementation
// ============================================================================

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, DataLoaderOptions options)
    : DataLoader(std::move(dataset), std::move(options), default_collate) {}

DataLoader::DataLoader(
    std::shared_ptr<Dataset> dataset,
    DataLoaderOptions options,
    CollateFn collate_fn
) : dataset_(std::move(dataset)),
    options_(std::move(options)),
    collate_fn_(std::move(collate_fn)),
    rng_(options_.seed == 0 ? std::random_device{}() : options_.seed) {

    if (options_.batch_size <= 0) {
        throw std::invalid_argument("batch_size must be positive");
    }

    // Initialize indices
    indices_.resize(dataset_->size());
    std::iota(indices_.begin(), indices_.end(), 0);

    if (options_.shuffle) {
        shuffle_indices();
    }

    // Start worker threads if num_workers > 0
    if (options_.num_workers > 0) {
        start_workers();
    }
}

DataLoader::~DataLoader() {
    stop_workers();
}

DataLoader::DataLoader(DataLoader&& other) noexcept
    : dataset_(std::move(other.dataset_)),
      options_(std::move(other.options_)),
      collate_fn_(std::move(other.collate_fn_)),
      indices_(std::move(other.indices_)),
      rng_(std::move(other.rng_)),
      epoch_(other.epoch_) {
    other.shutdown_ = true;
}

DataLoader& DataLoader::operator=(DataLoader&& other) noexcept {
    if (this != &other) {
        stop_workers();
        dataset_ = std::move(other.dataset_);
        options_ = std::move(other.options_);
        collate_fn_ = std::move(other.collate_fn_);
        indices_ = std::move(other.indices_);
        rng_ = std::move(other.rng_);
        epoch_ = other.epoch_;
        other.shutdown_ = true;
    }
    return *this;
}

void DataLoader::shuffle_indices() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

int64_t DataLoader::num_batches() const {
    int64_t n = dataset_->size();
    if (options_.drop_last) {
        return n / options_.batch_size;
    }
    return (n + options_.batch_size - 1) / options_.batch_size;
}

void DataLoader::reset_epoch() {
    ++epoch_;
    if (options_.shuffle) {
        shuffle_indices();
    }
}

Batch DataLoader::load_batch_impl(int64_t batch_idx) const {
    if (batch_idx < 0 || batch_idx >= num_batches()) {
        throw std::out_of_range("Batch index out of range: " +
            std::to_string(batch_idx) + " >= " + std::to_string(num_batches()));
    }

    int64_t start_idx = batch_idx * options_.batch_size;
    int64_t end_idx = std::min(start_idx + options_.batch_size,
                               static_cast<int64_t>(indices_.size()));

    // Collect samples for this batch
    std::vector<Sample> samples;
    samples.reserve(end_idx - start_idx);

    for (int64_t i = start_idx; i < end_idx; ++i) {
        samples.push_back(dataset_->get_item(indices_[i]));
    }

    return collate_fn_(samples);
}

Batch DataLoader::get_batch(int64_t batch_idx) const {
    return load_batch_impl(batch_idx);
}

void DataLoader::start_workers() {
    // Simplified: workers not fully implemented in this version
    // Real implementation would use a thread pool for prefetching
}

void DataLoader::stop_workers() {
    shutdown_ = true;
    queue_cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

// ============================================================================
// Iterator Implementation
// ============================================================================

DataLoader::Iterator::Iterator(DataLoader* loader, int64_t batch_idx)
    : loader_(loader), batch_idx_(batch_idx) {}

void DataLoader::Iterator::load_batch() const {
    if (!current_batch_.has_value() && loader_ && batch_idx_ < loader_->num_batches()) {
        current_batch_ = loader_->get_batch(batch_idx_);
    }
}

const Batch& DataLoader::Iterator::operator*() const {
    load_batch();
    return *current_batch_;
}

const Batch* DataLoader::Iterator::operator->() const {
    load_batch();
    return &(*current_batch_);
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    ++batch_idx_;
    current_batch_.reset();
    return *this;
}

DataLoader::Iterator DataLoader::Iterator::operator++(int) {
    Iterator tmp = *this;
    ++(*this);
    return tmp;
}

bool DataLoader::Iterator::operator==(const Iterator& other) const {
    return loader_ == other.loader_ && batch_idx_ == other.batch_idx_;
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return !(*this == other);
}

DataLoader::Iterator DataLoader::begin() {
    return Iterator(this, 0);
}

DataLoader::Iterator DataLoader::end() {
    return Iterator(this, num_batches());
}

}  // namespace pyflame::data
