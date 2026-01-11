#pragma once

#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include <optional>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>

#include "pyflame/data/dataset.hpp"
#include "pyflame/core/tensor.hpp"

namespace pyflame::data {

/// A batch of samples (stacked data and labels)
struct Batch {
    Tensor data;
    Tensor labels;

    Batch() = default;
    Batch(Tensor d, Tensor l) : data(std::move(d)), labels(std::move(l)) {}
};

/// Options for DataLoader configuration
struct DataLoaderOptions {
    int64_t batch_size = 1;           // Number of samples per batch
    bool shuffle = false;              // Shuffle data each epoch
    bool drop_last = false;            // Drop last incomplete batch
    int num_workers = 0;               // Number of worker threads (0 = main thread)
    uint64_t seed = 0;                 // Random seed (0 = random)
    bool pin_memory = false;           // Pin memory for faster GPU transfer
    std::optional<int64_t> prefetch_factor = std::nullopt;  // Batches to prefetch per worker
};

/// Collate function type - combines samples into a batch
using CollateFn = std::function<Batch(const std::vector<Sample>&)>;

/// Default collate function - stacks samples along dim 0
Batch default_collate(const std::vector<Sample>& samples);

/// DataLoader provides batched iteration over a dataset
class DataLoader {
public:
    /// Create DataLoader with options
    DataLoader(
        std::shared_ptr<Dataset> dataset,
        DataLoaderOptions options = {}
    );

    /// Create DataLoader with custom collate function
    DataLoader(
        std::shared_ptr<Dataset> dataset,
        DataLoaderOptions options,
        CollateFn collate_fn
    );

    ~DataLoader();

    // Non-copyable
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    // Movable
    DataLoader(DataLoader&&) noexcept;
    DataLoader& operator=(DataLoader&&) noexcept;

    /// Iterator class for range-based for loops
    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Batch;
        using difference_type = std::ptrdiff_t;
        using pointer = const Batch*;
        using reference = const Batch&;

        Iterator(DataLoader* loader, int64_t batch_idx);

        reference operator*() const;
        pointer operator->() const;
        Iterator& operator++();
        Iterator operator++(int);
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        DataLoader* loader_;
        int64_t batch_idx_;
        mutable std::optional<Batch> current_batch_;

        void load_batch() const;
    };

    /// Begin iteration (resets epoch)
    Iterator begin();

    /// End iterator
    Iterator end();

    /// Number of batches per epoch
    int64_t num_batches() const;

    /// Get a specific batch by index
    Batch get_batch(int64_t batch_idx) const;

    /// Reset for new epoch (reshuffles if shuffle=true)
    void reset_epoch();

    /// Get current epoch's sample indices
    const std::vector<int64_t>& indices() const { return indices_; }

    /// Get the underlying dataset
    std::shared_ptr<Dataset> dataset() const { return dataset_; }

    /// Get batch size
    int64_t batch_size() const { return options_.batch_size; }

    /// Get dataset size
    int64_t dataset_size() const { return dataset_->size(); }

private:
    std::shared_ptr<Dataset> dataset_;
    DataLoaderOptions options_;
    CollateFn collate_fn_;
    std::vector<int64_t> indices_;
    mutable std::mt19937 rng_;
    int64_t epoch_ = 0;

    // Worker thread state (for num_workers > 0)
    std::vector<std::thread> workers_;
    std::queue<std::future<Batch>> prefetch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_{false};

    void shuffle_indices();
    Batch load_batch_impl(int64_t batch_idx) const;
    void start_workers();
    void stop_workers();
};

// ============================================================================
// Samplers
// ============================================================================

/// Abstract base class for samplers
class Sampler {
public:
    virtual ~Sampler() = default;

    /// Get indices for iteration
    virtual std::vector<int64_t> get_indices() const = 0;

    /// Number of samples
    virtual int64_t size() const = 0;
};

/// Sequential sampler - returns indices in order
class SequentialSampler : public Sampler {
public:
    explicit SequentialSampler(int64_t size) : size_(size) {}

    std::vector<int64_t> get_indices() const override {
        std::vector<int64_t> indices(size_);
        for (int64_t i = 0; i < size_; ++i) {
            indices[i] = i;
        }
        return indices;
    }

    int64_t size() const override { return size_; }

private:
    int64_t size_;
};

/// Random sampler - returns shuffled indices
class RandomSampler : public Sampler {
public:
    RandomSampler(int64_t size, bool replacement = false, uint64_t seed = 0)
        : size_(size), replacement_(replacement), seed_(seed) {}

    std::vector<int64_t> get_indices() const override {
        std::mt19937 rng(seed_ == 0 ? std::random_device{}() : seed_);
        std::vector<int64_t> indices(size_);

        if (replacement_) {
            std::uniform_int_distribution<int64_t> dist(0, size_ - 1);
            for (int64_t i = 0; i < size_; ++i) {
                indices[i] = dist(rng);
            }
        } else {
            for (int64_t i = 0; i < size_; ++i) {
                indices[i] = i;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        return indices;
    }

    int64_t size() const override { return size_; }

private:
    int64_t size_;
    bool replacement_;
    uint64_t seed_;
};

/// Weighted random sampler
class WeightedRandomSampler : public Sampler {
public:
    WeightedRandomSampler(
        std::vector<double> weights,
        int64_t num_samples,
        bool replacement = true,
        uint64_t seed = 0
    ) : weights_(std::move(weights)),
        num_samples_(num_samples),
        replacement_(replacement),
        seed_(seed) {}

    std::vector<int64_t> get_indices() const override {
        std::mt19937 rng(seed_ == 0 ? std::random_device{}() : seed_);
        std::discrete_distribution<int64_t> dist(weights_.begin(), weights_.end());

        std::vector<int64_t> indices(num_samples_);
        for (int64_t i = 0; i < num_samples_; ++i) {
            indices[i] = dist(rng);
        }
        return indices;
    }

    int64_t size() const override { return num_samples_; }

private:
    std::vector<double> weights_;
    int64_t num_samples_;
    bool replacement_;
    uint64_t seed_;
};

}  // namespace pyflame::data
