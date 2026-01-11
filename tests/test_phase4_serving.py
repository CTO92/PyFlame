"""
Tests for PyFlame Phase 4 Serving Module.

Tests InferenceEngine, ModelServer, and Client.
"""

import pytest
import sys
import os
import time

# Add Python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pyflame.serving.inference import (
    InferenceEngine,
    InferenceConfig,
    InferenceStats,
    LRUCache,
    optimize_for_inference,
)
from pyflame.serving.client import (
    ModelClient,
    ClientConfig,
)


# =============================================================================
# Mock Classes
# =============================================================================

class MockModel:
    """Mock model for testing."""

    def __init__(self):
        self._eval_mode = False

    def __call__(self, x):
        import numpy as np
        if hasattr(x, "shape"):
            batch_size = x.shape[0] if len(x.shape) > 0 else 1
        else:
            batch_size = 1
        return np.random.randn(batch_size, 10).astype(np.float32)

    def eval(self):
        self._eval_mode = True

    def train(self):
        self._eval_mode = False


class MockTensor:
    """Mock tensor for testing."""

    def __init__(self, data):
        import numpy as np
        self._data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self._data.shape

    def numpy(self):
        return self._data


# =============================================================================
# InferenceConfig Tests
# =============================================================================

class TestInferenceConfig:
    """Test cases for InferenceConfig."""

    def test_default_config(self):
        config = InferenceConfig()
        assert config.max_batch_size == 32
        assert config.enable_caching is False
        assert config.cache_size == 1000
        assert config.warmup_iterations == 10

    def test_custom_config(self):
        config = InferenceConfig(
            max_batch_size=64,
            enable_caching=True,
            cache_size=500,
        )
        assert config.max_batch_size == 64
        assert config.enable_caching is True
        assert config.cache_size == 500


# =============================================================================
# InferenceStats Tests
# =============================================================================

class TestInferenceStats:
    """Test cases for InferenceStats."""

    def test_stats_initialization(self):
        stats = InferenceStats()
        assert stats.total_inferences == 0
        assert stats.total_time_ms == 0.0
        assert stats.cache_hits == 0

    def test_stats_average_time(self):
        stats = InferenceStats(total_inferences=10, total_time_ms=100.0)
        assert stats.average_time_ms == 10.0

    def test_stats_throughput(self):
        stats = InferenceStats(total_inferences=100, total_time_ms=1000.0)
        assert stats.throughput == 100.0

    def test_stats_cache_hit_rate(self):
        stats = InferenceStats(cache_hits=80, cache_misses=20)
        assert stats.cache_hit_rate == 0.8

    def test_stats_to_dict(self):
        stats = InferenceStats(
            total_inferences=50,
            total_time_ms=500.0,
        )
        d = stats.to_dict()
        assert "total_inferences" in d
        assert "average_time_ms" in d
        assert "throughput_per_second" in d


# =============================================================================
# LRUCache Tests
# =============================================================================

class TestLRUCache:
    """Test cases for LRUCache."""

    def test_cache_initialization(self):
        cache = LRUCache(max_size=10)
        assert len(cache) == 0

    def test_cache_put_get(self):
        cache = LRUCache(max_size=10)
        cache.put(1, "value1")
        cache.put(2, "value2")

        assert cache.get(1) == "value1"
        assert cache.get(2) == "value2"
        assert cache.get(3) is None

    def test_cache_eviction(self):
        cache = LRUCache(max_size=3)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.put(3, "c")
        cache.put(4, "d")  # Should evict key 1

        assert cache.get(1) is None  # Evicted
        assert cache.get(2) == "b"
        assert cache.get(4) == "d"

    def test_cache_lru_order(self):
        cache = LRUCache(max_size=3)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.put(3, "c")

        # Access key 1 to make it recently used
        cache.get(1)

        # Add new item, should evict key 2 (oldest)
        cache.put(4, "d")

        assert cache.get(1) == "a"  # Still present
        assert cache.get(2) is None  # Evicted
        assert cache.get(3) == "c"
        assert cache.get(4) == "d"

    def test_cache_clear(self):
        cache = LRUCache(max_size=10)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.clear()
        assert len(cache) == 0


# =============================================================================
# InferenceEngine Tests
# =============================================================================

class TestInferenceEngine:
    """Test cases for InferenceEngine."""

    def test_engine_initialization(self):
        model = MockModel()
        engine = InferenceEngine(model)

        assert engine.model is model
        assert engine._warmed_up is False

    def test_engine_infer(self):
        import numpy as np

        model = MockModel()
        engine = InferenceEngine(model)

        inputs = np.random.randn(4, 10).astype(np.float32)
        outputs = engine.infer(inputs)

        assert outputs is not None
        assert engine.stats.total_inferences == 1

    def test_engine_infer_with_time(self):
        import numpy as np

        model = MockModel()
        engine = InferenceEngine(model)

        inputs = np.random.randn(4, 10).astype(np.float32)
        outputs, time_ms = engine.infer(inputs, return_time=True)

        assert outputs is not None
        assert time_ms >= 0

    def test_engine_warmup(self):
        import numpy as np

        model = MockModel()
        engine = InferenceEngine(model)

        example_input = np.random.randn(1, 10).astype(np.float32)
        engine.warmup(example_input, num_iterations=5)

        assert engine._warmed_up is True

    def test_engine_with_caching(self):
        import numpy as np

        config = InferenceConfig(enable_caching=True, cache_size=100)
        model = MockModel()
        engine = InferenceEngine(model, config=config)

        inputs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        # First inference - cache miss
        engine.infer(inputs)
        assert engine.stats.cache_misses == 1

        # Second inference with same input - cache hit
        engine.infer(inputs)
        assert engine.stats.cache_hits == 1

    def test_engine_stats(self):
        import numpy as np

        model = MockModel()
        engine = InferenceEngine(model)

        for _ in range(5):
            inputs = np.random.randn(2, 10).astype(np.float32)
            engine.infer(inputs)

        assert engine.stats.total_inferences == 5
        assert engine.stats.average_time_ms > 0

    def test_engine_reset_stats(self):
        import numpy as np

        model = MockModel()
        engine = InferenceEngine(model)

        inputs = np.random.randn(2, 10).astype(np.float32)
        engine.infer(inputs)

        assert engine.stats.total_inferences == 1

        engine.reset_stats()
        assert engine.stats.total_inferences == 0

    def test_engine_get_info(self):
        model = MockModel()
        config = InferenceConfig(max_batch_size=64)
        engine = InferenceEngine(model, config=config)

        info = engine.get_info()
        assert info["config"]["max_batch_size"] == 64
        assert "stats" in info


# =============================================================================
# ModelClient Tests
# =============================================================================

class TestClientConfig:
    """Test cases for ClientConfig."""

    def test_default_config(self):
        config = ClientConfig()
        assert config.base_url == "http://localhost:8000"
        assert config.timeout == 30
        assert config.api_prefix == "/v1"

    def test_custom_config(self):
        config = ClientConfig(
            base_url="http://example.com:9000",
            timeout=60,
        )
        assert config.base_url == "http://example.com:9000"
        assert config.timeout == 60


class TestModelClient:
    """Test cases for ModelClient."""

    def test_client_initialization(self):
        client = ModelClient("http://localhost:8000")
        assert client.config.base_url == "http://localhost:8000"

    def test_client_build_url(self):
        client = ModelClient("http://localhost:8000")
        url = client._build_url("/health")
        assert url == "http://localhost:8000/v1/health"

    def test_client_with_config(self):
        config = ClientConfig(
            base_url="http://example.com",
            api_prefix="/api",
        )
        client = ModelClient(config=config)
        url = client._build_url("/predict")
        assert url == "http://example.com/api/predict"

    def test_client_context_manager(self):
        with ModelClient("http://localhost:8000") as client:
            assert client is not None


# =============================================================================
# optimize_for_inference Tests
# =============================================================================

class TestOptimizeForInference:
    """Test cases for optimize_for_inference function."""

    def test_optimize_basic(self):
        import numpy as np

        model = MockModel()
        example_input = np.random.randn(1, 10).astype(np.float32)

        optimized = optimize_for_inference(model, example_input)
        assert optimized is not None

    def test_optimize_with_options(self):
        import numpy as np

        model = MockModel()
        example_input = np.random.randn(1, 10).astype(np.float32)

        optimized = optimize_for_inference(
            model,
            example_input,
            fuse_operations=True,
            constant_folding=True,
            precision="fp32",
        )
        assert optimized is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
