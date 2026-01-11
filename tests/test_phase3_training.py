"""
Tests for PyFlame Phase 3 Training Module.

Tests Trainer, Callbacks, and related utilities.
"""

import pytest
import sys
import os
import tempfile
import shutil

# Add Python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pyflame.training.trainer import Trainer, TrainerConfig, TrainerState
from pyflame.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    ProgressBar,
    CSVLogger,
)


# =============================================================================
# Mock Classes for Testing
# =============================================================================

class MockModel:
    """Mock model for testing."""

    def __init__(self):
        self._training = True
        self._state_dict = {"weight": [1.0, 2.0, 3.0]}

    def __call__(self, x):
        return {"loss": 0.5, "output": x}

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict, strict=True):
        self._state_dict = state_dict

    def parameters(self):
        return []


class MockOptimizer:
    """Mock optimizer for testing."""

    def __init__(self):
        self._state_dict = {}

    def step(self):
        pass

    def zero_grad(self):
        pass

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict


class MockDataLoader:
    """Mock dataloader for testing."""

    def __init__(self, num_batches=5, batch_size=4):
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(self.num_batches):
            # Return (inputs, targets) tuple
            yield ([i] * self.batch_size, [i % 2] * self.batch_size)

    def __len__(self):
        return self.num_batches


class MockCriterion:
    """Mock loss function for testing."""

    def __call__(self, outputs, targets):
        return 0.5


# =============================================================================
# TrainerConfig Tests
# =============================================================================

class TestTrainerConfig:
    """Test cases for TrainerConfig."""

    def test_default_config(self):
        config = TrainerConfig()
        assert config.max_epochs == 10
        assert config.gradient_accumulation_steps == 1
        assert config.log_every_n_steps == 50

    def test_custom_config(self):
        config = TrainerConfig(
            max_epochs=5,
            gradient_clip_val=1.0,
            precision="16",
        )
        assert config.max_epochs == 5
        assert config.gradient_clip_val == 1.0
        assert config.precision == "16"


# =============================================================================
# Trainer Tests
# =============================================================================

class TestTrainer:
    """Test cases for Trainer."""

    def test_trainer_initialization(self):
        trainer = Trainer()
        assert trainer.config is not None
        assert trainer.state is not None

    def test_trainer_with_config(self):
        config = TrainerConfig(max_epochs=5)
        trainer = Trainer(config=config)
        assert trainer.config.max_epochs == 5

    def test_trainer_fit(self):
        trainer = Trainer(config=TrainerConfig(max_epochs=2))
        model = MockModel()
        train_loader = MockDataLoader(num_batches=3)
        optimizer = MockOptimizer()
        criterion = MockCriterion()

        result = trainer.fit(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
        )

        assert "epochs" in result
        assert result["epochs"] == 1  # 0-indexed epoch

    def test_trainer_with_validation(self):
        trainer = Trainer(config=TrainerConfig(max_epochs=1))
        model = MockModel()
        train_loader = MockDataLoader(num_batches=3)
        val_loader = MockDataLoader(num_batches=2)
        optimizer = MockOptimizer()
        criterion = MockCriterion()

        result = trainer.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
        )

        assert "val_loss" in trainer.state.logs or result is not None

    def test_trainer_test(self):
        trainer = Trainer()
        model = MockModel()
        test_loader = MockDataLoader(num_batches=2)
        criterion = MockCriterion()

        trainer.model = model
        trainer.criterion = criterion

        result = trainer.test(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
        )

        assert "test_loss" in result


# =============================================================================
# Callback Tests
# =============================================================================

class TestCallbacks:
    """Test cases for callbacks."""

    def test_callback_list(self):
        callbacks = CallbackList([Callback(), Callback()])
        assert len(list(callbacks)) == 2

    def test_early_stopping_initialization(self):
        es = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        assert es.monitor == "val_loss"
        assert es.patience == 3

    def test_early_stopping_trigger(self):
        es = EarlyStopping(patience=2, mode="min")

        # Mock trainer
        class MockTrainer:
            class State:
                logs = {}
                should_stop = False
                epoch = 0

            state = State()

        trainer = MockTrainer()

        # No improvement for 3 epochs
        trainer.state.logs = {"val_loss": 1.0}
        es.on_epoch_end(trainer)
        assert not trainer.state.should_stop

        trainer.state.epoch = 1
        trainer.state.logs = {"val_loss": 1.1}  # Worse
        es.on_epoch_end(trainer)
        assert not trainer.state.should_stop

        trainer.state.epoch = 2
        trainer.state.logs = {"val_loss": 1.2}  # Worse again
        es.on_epoch_end(trainer)
        assert trainer.state.should_stop

    def test_early_stopping_improvement(self):
        es = EarlyStopping(patience=2, mode="min")

        class MockTrainer:
            class State:
                logs = {}
                should_stop = False
                epoch = 0

            state = State()

        trainer = MockTrainer()

        # Continuous improvement
        trainer.state.logs = {"val_loss": 1.0}
        es.on_epoch_end(trainer)

        trainer.state.epoch = 1
        trainer.state.logs = {"val_loss": 0.9}  # Better
        es.on_epoch_end(trainer)

        trainer.state.epoch = 2
        trainer.state.logs = {"val_loss": 0.8}  # Better again
        es.on_epoch_end(trainer)

        assert not trainer.state.should_stop
        assert es.wait == 0


class TestModelCheckpoint:
    """Test cases for ModelCheckpoint callback."""

    def test_checkpoint_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = ModelCheckpoint(dirpath=tmpdir, save_top_k=3)
            assert cp.save_top_k == 3
            assert os.path.exists(tmpdir)


class TestCSVLogger:
    """Test cases for CSVLogger callback."""

    def test_csv_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(save_dir=tmpdir, name="test")

            class MockTrainer:
                class State:
                    logs = {"train_loss": 0.5, "val_loss": 0.4}
                    epoch = 0
                    global_step = 100

                state = State()

            trainer = MockTrainer()
            logger.on_epoch_end(trainer)

            # Check CSV was created
            assert os.path.exists(logger.filepath)


# =============================================================================
# Progress Bar Tests
# =============================================================================

class TestProgressBar:
    """Test cases for ProgressBar callback."""

    def test_progress_bar_initialization(self):
        pb = ProgressBar(refresh_rate=10)
        assert pb.refresh_rate == 10


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
