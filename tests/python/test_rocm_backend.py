"""
Tests for ROCm backend functionality.

These tests verify that the ROCm backend produces correct results by
comparing against CPU reference implementations.
"""

import pytest
import numpy as np

import pyflame as pf


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def rocm_device():
    """Fixture to set up ROCm device for tests."""
    if not pf.rocm_is_available():
        pytest.skip("ROCm not available")
    original_device = pf.get_device()
    pf.set_device('rocm')
    yield
    pf.set_device(original_device)  # Reset after test


@pytest.fixture
def cpu_device():
    """Fixture to use CPU device."""
    original_device = pf.get_device()
    pf.set_device('cpu')
    yield
    pf.set_device(original_device)


# =============================================================================
# Basic Backend Tests
# =============================================================================

class TestROCmBackend:
    """Tests for basic ROCm backend functionality."""

    def test_rocm_available(self):
        """Test ROCm availability check."""
        result = pf.rocm_is_available()
        assert isinstance(result, bool)

    def test_device_selection(self, rocm_device):
        """Test device selection."""
        device = pf.get_device()
        assert device.startswith('rocm') or device == 'rocm'

    def test_device_info(self, rocm_device):
        """Test device info retrieval."""
        info = pf.device_info()
        assert 'name' in info
        assert 'total_memory' in info
        assert info['type'] == 'rocm'
        assert info['total_memory'] > 0

    def test_synchronize(self, rocm_device):
        """Test device synchronization."""
        # Should not raise
        pf.synchronize()

    def test_device_count(self):
        """Test device count retrieval."""
        if not pf.rocm_is_available():
            pytest.skip("ROCm not available")
        count = pf.device_count()
        assert count >= 1


# =============================================================================
# Matrix Operation Tests
# =============================================================================

class TestROCmMatmul:
    """Tests for matrix multiplication on ROCm."""

    def test_matmul_small(self, rocm_device):
        """Test small matrix multiplication."""
        np.random.seed(42)
        a_np = np.random.randn(16, 32).astype(np.float32)
        b_np = np.random.randn(32, 24).astype(np.float32)

        a = pf.tensor(a_np)
        b = pf.tensor(b_np)

        # ROCm result
        pf.set_device('rocm')
        c_rocm = pf.matmul(a, b)

        # CPU reference
        pf.set_device('cpu')
        c_cpu = pf.matmul(a, b)

        np.testing.assert_allclose(
            c_rocm.numpy(), c_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_matmul_medium(self, rocm_device):
        """Test medium matrix multiplication."""
        np.random.seed(42)
        a_np = np.random.randn(128, 256).astype(np.float32)
        b_np = np.random.randn(256, 512).astype(np.float32)

        a = pf.tensor(a_np)
        b = pf.tensor(b_np)

        pf.set_device('rocm')
        c_rocm = pf.matmul(a, b)

        pf.set_device('cpu')
        c_cpu = pf.matmul(a, b)

        np.testing.assert_allclose(
            c_rocm.numpy(), c_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_matmul_large(self, rocm_device):
        """Test large matrix multiplication."""
        np.random.seed(42)
        a_np = np.random.randn(512, 1024).astype(np.float32)
        b_np = np.random.randn(1024, 512).astype(np.float32)

        a = pf.tensor(a_np)
        b = pf.tensor(b_np)

        pf.set_device('rocm')
        c_rocm = pf.matmul(a, b)

        pf.set_device('cpu')
        c_cpu = pf.matmul(a, b)

        np.testing.assert_allclose(
            c_rocm.numpy(), c_cpu.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_batched_matmul(self, rocm_device):
        """Test batched matrix multiplication."""
        np.random.seed(42)
        a_np = np.random.randn(4, 32, 64).astype(np.float32)
        b_np = np.random.randn(4, 64, 48).astype(np.float32)

        a = pf.tensor(a_np)
        b = pf.tensor(b_np)

        pf.set_device('rocm')
        c_rocm = pf.bmm(a, b)

        pf.set_device('cpu')
        c_cpu = pf.bmm(a, b)

        np.testing.assert_allclose(
            c_rocm.numpy(), c_cpu.numpy(), rtol=1e-5, atol=1e-6
        )


# =============================================================================
# Convolution Tests
# =============================================================================

class TestROCmConv2d:
    """Tests for 2D convolution on ROCm."""

    def test_conv2d_basic(self, rocm_device):
        """Test basic 2D convolution."""
        np.random.seed(42)
        x_np = np.random.randn(2, 3, 32, 32).astype(np.float32)
        w_np = np.random.randn(16, 3, 3, 3).astype(np.float32)

        x = pf.tensor(x_np)
        w = pf.tensor(w_np)

        pf.set_device('rocm')
        y_rocm = pf.conv2d(x, w)

        pf.set_device('cpu')
        y_cpu = pf.conv2d(x, w)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_conv2d_with_padding(self, rocm_device):
        """Test 2D convolution with padding."""
        np.random.seed(42)
        x_np = np.random.randn(4, 16, 28, 28).astype(np.float32)
        w_np = np.random.randn(32, 16, 3, 3).astype(np.float32)

        x = pf.tensor(x_np)
        w = pf.tensor(w_np)

        pf.set_device('rocm')
        y_rocm = pf.conv2d(x, w, padding=1)

        pf.set_device('cpu')
        y_cpu = pf.conv2d(x, w, padding=1)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_conv2d_with_stride(self, rocm_device):
        """Test 2D convolution with stride."""
        np.random.seed(42)
        x_np = np.random.randn(2, 32, 64, 64).astype(np.float32)
        w_np = np.random.randn(64, 32, 3, 3).astype(np.float32)

        x = pf.tensor(x_np)
        w = pf.tensor(w_np)

        pf.set_device('rocm')
        y_rocm = pf.conv2d(x, w, stride=2, padding=1)

        pf.set_device('cpu')
        y_cpu = pf.conv2d(x, w, stride=2, padding=1)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-4, atol=1e-5
        )


# =============================================================================
# Pooling Tests
# =============================================================================

class TestROCmPooling:
    """Tests for pooling operations on ROCm."""

    def test_max_pool2d(self, rocm_device):
        """Test 2D max pooling."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 32, 32).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.max_pool2d(x, kernel_size=2, stride=2)

        pf.set_device('cpu')
        y_cpu = pf.max_pool2d(x, kernel_size=2, stride=2)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_avg_pool2d(self, rocm_device):
        """Test 2D average pooling."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 32, 32).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.avg_pool2d(x, kernel_size=2, stride=2)

        pf.set_device('cpu')
        y_cpu = pf.avg_pool2d(x, kernel_size=2, stride=2)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_global_avg_pool2d(self, rocm_device):
        """Test global average pooling."""
        np.random.seed(42)
        x_np = np.random.randn(4, 32, 7, 7).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.adaptive_avg_pool2d(x, (1, 1))

        pf.set_device('cpu')
        y_cpu = pf.adaptive_avg_pool2d(x, (1, 1))

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )


# =============================================================================
# Activation Function Tests
# =============================================================================

class TestROCmActivations:
    """Tests for activation functions on ROCm."""

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh", "gelu", "silu"])
    def test_activations(self, rocm_device, activation):
        """Test activation functions."""
        np.random.seed(42)
        x_np = np.random.randn(1024).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = getattr(pf, activation)(x)

        pf.set_device('cpu')
        y_cpu = getattr(pf, activation)(x)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_leaky_relu(self, rocm_device):
        """Test leaky ReLU."""
        np.random.seed(42)
        x_np = np.random.randn(1024).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.leaky_relu(x, negative_slope=0.01)

        pf.set_device('cpu')
        y_cpu = pf.leaky_relu(x, negative_slope=0.01)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_softmax(self, rocm_device):
        """Test softmax."""
        np.random.seed(42)
        x_np = np.random.randn(32, 10).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.softmax(x, dim=-1)

        pf.set_device('cpu')
        y_cpu = pf.softmax(x, dim=-1)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_log_softmax(self, rocm_device):
        """Test log softmax."""
        np.random.seed(42)
        x_np = np.random.randn(32, 10).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.log_softmax(x, dim=-1)

        pf.set_device('cpu')
        y_cpu = pf.log_softmax(x, dim=-1)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )


# =============================================================================
# Elementwise Operation Tests
# =============================================================================

class TestROCmElementwise:
    """Tests for elementwise operations on ROCm."""

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_binary_ops(self, rocm_device, op):
        """Test elementwise binary operations."""
        np.random.seed(42)
        a_np = np.random.randn(4096).astype(np.float32)
        b_np = np.random.randn(4096).astype(np.float32)
        if op == "div":
            b_np = np.abs(b_np) + 0.5  # Avoid division by zero

        a = pf.tensor(a_np)
        b = pf.tensor(b_np)

        pf.set_device('rocm')
        if op == "add":
            y_rocm = a + b
        elif op == "sub":
            y_rocm = a - b
        elif op == "mul":
            y_rocm = a * b
        elif op == "div":
            y_rocm = a / b

        pf.set_device('cpu')
        if op == "add":
            y_cpu = a + b
        elif op == "sub":
            y_cpu = a - b
        elif op == "mul":
            y_cpu = a * b
        elif op == "div":
            y_cpu = a / b

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize("op", ["neg", "abs", "sqrt", "exp", "log"])
    def test_unary_ops(self, rocm_device, op):
        """Test elementwise unary operations."""
        np.random.seed(42)
        if op in ["sqrt", "log"]:
            x_np = np.abs(np.random.randn(4096).astype(np.float32)) + 0.1
        elif op == "exp":
            x_np = np.random.randn(4096).astype(np.float32) * 0.5  # Avoid overflow
        else:
            x_np = np.random.randn(4096).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        if op == "neg":
            y_rocm = -x
        else:
            y_rocm = getattr(pf, op)(x)

        pf.set_device('cpu')
        if op == "neg":
            y_cpu = -x
        else:
            y_cpu = getattr(pf, op)(x)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )


# =============================================================================
# Reduction Operation Tests
# =============================================================================

class TestROCmReductions:
    """Tests for reduction operations on ROCm."""

    def test_sum(self, rocm_device):
        """Test sum reduction."""
        np.random.seed(42)
        x_np = np.random.randn(32, 64).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.sum(x, dim=1)

        pf.set_device('cpu')
        y_cpu = pf.sum(x, dim=1)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_mean(self, rocm_device):
        """Test mean reduction."""
        np.random.seed(42)
        x_np = np.random.randn(32, 64).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.mean(x, dim=1)

        pf.set_device('cpu')
        y_cpu = pf.mean(x, dim=1)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_max(self, rocm_device):
        """Test max reduction."""
        np.random.seed(42)
        x_np = np.random.randn(32, 64).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.max(x, dim=1)

        pf.set_device('cpu')
        y_cpu = pf.max(x, dim=1)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_min(self, rocm_device):
        """Test min reduction."""
        np.random.seed(42)
        x_np = np.random.randn(32, 64).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.min(x, dim=1)

        pf.set_device('cpu')
        y_cpu = pf.min(x, dim=1)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )


# =============================================================================
# Loss Function Tests
# =============================================================================

class TestROCmLosses:
    """Tests for loss functions on ROCm."""

    def test_mse_loss(self, rocm_device):
        """Test MSE loss."""
        np.random.seed(42)
        pred_np = np.random.randn(32, 10).astype(np.float32)
        target_np = np.random.randn(32, 10).astype(np.float32)

        pred = pf.tensor(pred_np)
        target = pf.tensor(target_np)

        pf.set_device('rocm')
        loss_rocm = pf.mse_loss(pred, target)

        pf.set_device('cpu')
        loss_cpu = pf.mse_loss(pred, target)

        np.testing.assert_allclose(
            loss_rocm.numpy(), loss_cpu.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_bce_loss(self, rocm_device):
        """Test BCE loss."""
        np.random.seed(42)
        pred_np = np.random.uniform(0.01, 0.99, (32, 2)).astype(np.float32)
        target_np = np.random.randint(0, 2, (32, 2)).astype(np.float32)

        pred = pf.tensor(pred_np)
        target = pf.tensor(target_np)

        pf.set_device('rocm')
        loss_rocm = pf.bce_loss(pred, target)

        pf.set_device('cpu')
        loss_cpu = pf.bce_loss(pred, target)

        np.testing.assert_allclose(
            loss_rocm.numpy(), loss_cpu.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_cross_entropy_loss(self, rocm_device):
        """Test cross-entropy loss."""
        np.random.seed(42)
        logits_np = np.random.randn(32, 10).astype(np.float32)
        targets_np = np.random.randint(0, 10, (32,)).astype(np.int64)

        logits = pf.tensor(logits_np)
        targets = pf.tensor(targets_np)

        pf.set_device('rocm')
        loss_rocm = pf.cross_entropy(logits, targets)

        pf.set_device('cpu')
        loss_cpu = pf.cross_entropy(logits, targets)

        np.testing.assert_allclose(
            loss_rocm.numpy(), loss_cpu.numpy(), rtol=1e-4, atol=1e-5
        )


# =============================================================================
# Memory Tests
# =============================================================================

class TestROCmMemory:
    """Tests for ROCm memory management."""

    def test_memory_info(self, rocm_device):
        """Test memory info retrieval."""
        info = pf.device_info()
        assert info['total_memory'] > 0
        assert info['free_memory'] > 0
        assert info['free_memory'] <= info['total_memory']

    def test_large_allocation(self, rocm_device):
        """Test allocation of large tensors."""
        info = pf.device_info()
        # Skip if not enough memory
        required = 256 * 1024 * 1024 * 4  # 1GB
        if info['free_memory'] < required:
            pytest.skip("Not enough GPU memory for large allocation test")

        # Allocate 1GB tensor
        size = 256 * 1024 * 1024  # 256M floats = 1GB
        x = pf.zeros(size)
        assert x.numel() == size

        # Verify by doing a simple operation
        y = x + 1
        assert y.sum().item() == size


# =============================================================================
# Complex Graph Tests
# =============================================================================

class TestROCmComplexGraphs:
    """Tests for complex computation graphs on ROCm."""

    def test_simple_mlp(self, rocm_device):
        """Test a simple MLP forward pass."""
        np.random.seed(42)

        x_np = np.random.randn(32, 784).astype(np.float32)
        w1_np = np.random.randn(784, 256).astype(np.float32)
        b1_np = np.random.randn(256).astype(np.float32)
        w2_np = np.random.randn(256, 10).astype(np.float32)
        b2_np = np.random.randn(10).astype(np.float32)

        x = pf.tensor(x_np)
        w1 = pf.tensor(w1_np)
        b1 = pf.tensor(b1_np)
        w2 = pf.tensor(w2_np)
        b2 = pf.tensor(b2_np)

        def mlp_forward(x, w1, b1, w2, b2):
            h = pf.relu(pf.matmul(x, w1) + b1)
            return pf.matmul(h, w2) + b2

        pf.set_device('rocm')
        y_rocm = mlp_forward(x, w1, b1, w2, b2)

        pf.set_device('cpu')
        y_cpu = mlp_forward(x, w1, b1, w2, b2)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_conv_block(self, rocm_device):
        """Test a conv-bn-relu block."""
        np.random.seed(42)

        x_np = np.random.randn(4, 32, 14, 14).astype(np.float32)
        conv_w_np = np.random.randn(64, 32, 3, 3).astype(np.float32)

        x = pf.tensor(x_np)
        conv_w = pf.tensor(conv_w_np)

        def conv_block(x, w):
            y = pf.conv2d(x, w, padding=1)
            return pf.relu(y)

        pf.set_device('rocm')
        y_rocm = conv_block(x, conv_w)

        pf.set_device('cpu')
        y_cpu = conv_block(x, conv_w)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-4, atol=1e-5
        )


# =============================================================================
# Edge Cases
# =============================================================================

class TestROCmEdgeCases:
    """Tests for edge cases on ROCm."""

    def test_empty_tensor(self, rocm_device):
        """Test operations on empty tensors."""
        x = pf.zeros(0)
        # Should not crash
        y = pf.relu(x)
        assert y.numel() == 0

    def test_single_element(self, rocm_device):
        """Test operations on single-element tensors."""
        np.random.seed(42)
        x_np = np.array([3.14159], dtype=np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.relu(x)

        pf.set_device('cpu')
        y_cpu = pf.relu(x)

        np.testing.assert_allclose(y_rocm.numpy(), y_cpu.numpy())

    def test_large_tensor(self, rocm_device):
        """Test operations on large tensors."""
        info = pf.device_info()
        # Skip if not enough memory
        required = 64 * 1024 * 1024 * 4  # 256MB
        if info['free_memory'] < required:
            pytest.skip("Not enough GPU memory")

        np.random.seed(42)
        size = 4 * 1024 * 1024  # 4M floats = 16MB
        x_np = np.random.randn(size).astype(np.float32)

        x = pf.tensor(x_np)

        pf.set_device('rocm')
        y_rocm = pf.relu(x)

        pf.set_device('cpu')
        y_cpu = pf.relu(x)

        np.testing.assert_allclose(
            y_rocm.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
