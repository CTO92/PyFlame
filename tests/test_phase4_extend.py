"""
Tests for PyFlame Phase 4 Extend Module.

Tests custom operator registration and plugin system.
"""

import pytest
import sys
import os

# Add Python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pyflame.extend.custom_op import (
    CustomOp,
    register_custom_op,
    custom_op,
    get_custom_op,
    list_custom_ops,
    unregister_custom_op,
    clear_custom_ops,
    AutogradFunction,
    _FunctionContext,
)
from pyflame.extend.plugin import (
    Plugin,
    PluginManager,
    PluginInfo,
    load_plugin,
    unload_plugin,
    list_plugins,
    register_plugin,
    get_plugin,
)


# =============================================================================
# CustomOp Tests
# =============================================================================

class TestCustomOp:
    """Test cases for CustomOp class."""

    def test_custom_op_creation(self):
        def my_forward(x):
            return x * 2

        op = CustomOp(
            name="test_op",
            forward_fn=my_forward,
        )

        assert op.name == "test_op"
        assert op.forward_fn is my_forward
        assert op.backward_fn is None

    def test_custom_op_call(self):
        def my_forward(x):
            return x * 2

        op = CustomOp(name="test_op", forward_fn=my_forward)
        result = op(5)

        assert result == 10

    def test_custom_op_repr(self):
        op = CustomOp(name="my_op", forward_fn=lambda x: x)
        assert "my_op" in repr(op)


# =============================================================================
# register_custom_op Tests
# =============================================================================

class TestRegisterCustomOp:
    """Test cases for register_custom_op function."""

    def setup_method(self):
        """Clear custom ops before each test."""
        clear_custom_ops()

    def teardown_method(self):
        """Clear custom ops after each test."""
        clear_custom_ops()

    def test_register_custom_op(self):
        def my_activation(x):
            return x if x > 0 else 0

        op = register_custom_op("test_relu", my_activation)

        assert op.name == "test_relu"
        assert "test_relu" in list_custom_ops()

    def test_register_custom_op_with_backward(self):
        def forward(x):
            return x ** 2

        def backward(grad_output, x):
            return grad_output * 2 * x

        op = register_custom_op(
            "test_square",
            forward_fn=forward,
            backward_fn=backward,
        )

        assert op.backward_fn is backward

    def test_register_duplicate_raises_error(self):
        def func(x):
            return x

        register_custom_op("duplicate_op", func)

        with pytest.raises(ValueError):
            register_custom_op("duplicate_op", func)

    def test_get_custom_op(self):
        def func(x):
            return x * 3

        register_custom_op("get_test_op", func)
        op = get_custom_op("get_test_op")

        assert op is not None
        assert op.name == "get_test_op"

    def test_get_nonexistent_op(self):
        op = get_custom_op("nonexistent_op")
        assert op is None

    def test_unregister_custom_op(self):
        def func(x):
            return x

        register_custom_op("unregister_test", func)
        assert "unregister_test" in list_custom_ops()

        result = unregister_custom_op("unregister_test")
        assert result is True
        assert "unregister_test" not in list_custom_ops()

    def test_unregister_nonexistent_op(self):
        result = unregister_custom_op("nonexistent")
        assert result is False

    def test_clear_custom_ops(self):
        register_custom_op("op1", lambda x: x)
        register_custom_op("op2", lambda x: x)

        assert len(list_custom_ops()) >= 2

        clear_custom_ops()
        assert len(list_custom_ops()) == 0


# =============================================================================
# custom_op Decorator Tests
# =============================================================================

class TestCustomOpDecorator:
    """Test cases for @custom_op decorator."""

    def setup_method(self):
        clear_custom_ops()

    def teardown_method(self):
        clear_custom_ops()

    def test_custom_op_decorator(self):
        @custom_op("decorated_op")
        def my_op(x):
            return x + 1

        assert "decorated_op" in list_custom_ops()
        assert my_op(5) == 6

    def test_custom_op_decorator_with_schema(self):
        @custom_op("schema_op", schema="(Tensor x) -> Tensor")
        def my_op(x):
            return x * 2

        op = get_custom_op("schema_op")
        assert op.schema == "(Tensor x) -> Tensor"


# =============================================================================
# AutogradFunction Tests
# =============================================================================

class TestAutogradFunction:
    """Test cases for AutogradFunction."""

    def test_function_context(self):
        ctx = _FunctionContext()
        ctx.save_for_backward(1, 2, 3)

        assert ctx.saved_tensors == (1, 2, 3)

    def test_function_context_attributes(self):
        ctx = _FunctionContext()
        ctx.beta = 1.5
        ctx.gamma = 2.0

        assert ctx.beta == 1.5
        assert ctx.gamma == 2.0


# =============================================================================
# Plugin Tests
# =============================================================================

class TestPluginInfo:
    """Test cases for PluginInfo."""

    def test_plugin_info_creation(self):
        info = PluginInfo(
            name="test-plugin",
            version="1.0.0",
            description="A test plugin",
        )

        assert info.name == "test-plugin"
        assert info.version == "1.0.0"
        assert info.description == "A test plugin"


class TestPlugin:
    """Test cases for Plugin base class."""

    def test_plugin_subclass(self):
        class MyPlugin(Plugin):
            name = "my-plugin"
            version = "1.0.0"

            def setup(self):
                pass

            def teardown(self):
                pass

        plugin = MyPlugin()
        assert plugin.name == "my-plugin"
        assert plugin._initialized is False

    def test_plugin_get_info(self):
        class MyPlugin(Plugin):
            name = "info-plugin"
            version = "2.0.0"
            description = "Test description"

            def setup(self):
                pass

            def teardown(self):
                pass

        plugin = MyPlugin()
        info = plugin.get_info()

        assert isinstance(info, PluginInfo)
        assert info.name == "info-plugin"
        assert info.version == "2.0.0"

    def test_plugin_get_custom_ops(self):
        class MyPlugin(Plugin):
            name = "ops-plugin"

            def setup(self):
                pass

            def teardown(self):
                pass

            def get_custom_ops(self):
                return {"my_op": lambda x: x * 2}

        plugin = MyPlugin()
        ops = plugin.get_custom_ops()

        assert "my_op" in ops
        assert ops["my_op"](5) == 10


# =============================================================================
# PluginManager Tests
# =============================================================================

class TestPluginManager:
    """Test cases for PluginManager."""

    def setup_method(self):
        # Get fresh instance
        PluginManager._instance = None

    def test_plugin_manager_singleton(self):
        manager1 = PluginManager.get_instance()
        manager2 = PluginManager.get_instance()

        assert manager1 is manager2

    def test_register_plugin_class(self):
        class TestPlugin(Plugin):
            name = "registered-plugin"

            def setup(self):
                pass

            def teardown(self):
                pass

        manager = PluginManager.get_instance()
        manager.register(TestPlugin)

        assert "registered-plugin" in manager.list_available()

    def test_load_registered_plugin(self):
        class LoadablePlugin(Plugin):
            name = "loadable-plugin"
            setup_called = False

            def setup(self):
                LoadablePlugin.setup_called = True

            def teardown(self):
                pass

        manager = PluginManager.get_instance()
        manager.register(LoadablePlugin)

        plugin = manager.load("loadable-plugin")

        assert plugin.name == "loadable-plugin"
        assert plugin._initialized is True
        assert LoadablePlugin.setup_called is True

    def test_unload_plugin(self):
        class UnloadablePlugin(Plugin):
            name = "unloadable-plugin"
            teardown_called = False

            def setup(self):
                pass

            def teardown(self):
                UnloadablePlugin.teardown_called = True

        manager = PluginManager.get_instance()
        manager.register(UnloadablePlugin)
        manager.load("unloadable-plugin")

        assert manager.get("unloadable-plugin") is not None

        manager.unload("unloadable-plugin")

        assert UnloadablePlugin.teardown_called is True
        assert manager.get("unloadable-plugin") is None

    def test_list_plugins(self):
        class ListablePlugin(Plugin):
            name = "listable-plugin"
            version = "1.0.0"
            description = "A listable plugin"

            def setup(self):
                pass

            def teardown(self):
                pass

        manager = PluginManager.get_instance()
        manager.register(ListablePlugin)
        manager.load("listable-plugin")

        plugins = manager.list_plugins()
        names = [p["name"] for p in plugins]

        assert "listable-plugin" in names


# =============================================================================
# Global Plugin Functions Tests
# =============================================================================

class TestGlobalPluginFunctions:
    """Test cases for global plugin functions."""

    def setup_method(self):
        PluginManager._instance = None

    def test_register_plugin_decorator(self):
        @register_plugin
        class DecoratedPlugin(Plugin):
            name = "decorated-plugin"

            def setup(self):
                pass

            def teardown(self):
                pass

        available = PluginManager.get_instance().list_available()
        assert "decorated-plugin" in available

    def test_load_plugin_function(self):
        @register_plugin
        class FunctionLoadPlugin(Plugin):
            name = "function-load-plugin"

            def setup(self):
                pass

            def teardown(self):
                pass

        plugin = load_plugin("function-load-plugin")
        assert plugin.name == "function-load-plugin"

    def test_get_plugin_function(self):
        @register_plugin
        class GetPlugin(Plugin):
            name = "get-plugin"

            def setup(self):
                pass

            def teardown(self):
                pass

        load_plugin("get-plugin")
        plugin = get_plugin("get-plugin")

        assert plugin is not None
        assert plugin.name == "get-plugin"

    def test_list_plugins_function(self):
        @register_plugin
        class ListPlugin(Plugin):
            name = "list-plugin"

            def setup(self):
                pass

            def teardown(self):
                pass

        load_plugin("list-plugin")
        plugins = list_plugins()

        names = [p["name"] for p in plugins]
        assert "list-plugin" in names


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
