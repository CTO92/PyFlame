"""
Plugin system for PyFlame extensions.

Allows third-party packages to extend PyFlame functionality.
"""

from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import importlib
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Security: Whitelist of allowed plugin module prefixes
# Plugins must be from these namespaces to be dynamically loaded
ALLOWED_PLUGIN_PREFIXES = frozenset({
    'pyflame.',
    'pyflame_',
    'pyflame_plugins.',
})

# Environment variable to add additional allowed prefixes (comma-separated)
_extra_prefixes = os.environ.get('PYFLAME_ALLOWED_PLUGIN_PREFIXES', '')
if _extra_prefixes:
    ALLOWED_PLUGIN_PREFIXES = ALLOWED_PLUGIN_PREFIXES | frozenset(_extra_prefixes.split(','))


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


@dataclass
class PluginInfo:
    """Plugin metadata.

    Attributes:
        name: Plugin name
        version: Plugin version
        description: Plugin description
        author: Plugin author
        dependencies: Required plugin dependencies
        tags: Plugin tags
    """
    name: str
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class Plugin(ABC):
    """Base class for PyFlame plugins.

    Plugins can extend PyFlame with:
    - Custom operators
    - New backends
    - Data loaders
    - Integrations
    - Model architectures

    Example:
        >>> class MyPlugin(Plugin):
        ...     name = "my-plugin"
        ...     version = "1.0.0"
        ...     description = "My custom plugin"
        ...
        ...     def setup(self):
        ...         # Register custom operators
        ...         from pyflame.extend import register_custom_op
        ...         register_custom_op("my_op", self._my_op_impl)
        ...
        ...     def teardown(self):
        ...         pass
        ...
        ...     def _my_op_impl(self, x):
        ...         return x * 2
    """

    # Plugin metadata (override in subclasses)
    name: str = "unnamed-plugin"
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = []
    tags: List[str] = []

    def __init__(self):
        """Initialize plugin."""
        self._initialized = False

    @abstractmethod
    def setup(self):
        """Initialize the plugin.

        Called when the plugin is loaded. Register custom operators,
        integrations, and other extensions here.
        """
        pass

    @abstractmethod
    def teardown(self):
        """Cleanup the plugin.

        Called when the plugin is unloaded. Clean up any resources,
        unregister operators, etc.
        """
        pass

    def get_info(self) -> PluginInfo:
        """Get plugin information.

        Returns:
            PluginInfo instance
        """
        return PluginInfo(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            dependencies=self.dependencies,
            tags=self.tags,
        )

    def get_custom_ops(self) -> Dict[str, Callable]:
        """Return custom operators provided by this plugin.

        Override to provide custom operators.

        Returns:
            Dictionary mapping operator names to functions
        """
        return {}

    def get_models(self) -> Dict[str, Type]:
        """Return model classes provided by this plugin.

        Override to provide custom model architectures.

        Returns:
            Dictionary mapping model names to classes
        """
        return {}

    def get_integrations(self) -> Dict[str, Any]:
        """Return integrations provided by this plugin.

        Returns:
            Dictionary of integration components
        """
        return {}


class PluginManager:
    """Manager for PyFlame plugins.

    Handles plugin discovery, loading, and lifecycle.

    Example:
        >>> manager = PluginManager()
        >>> manager.load("my-plugin")
        >>> manager.list_plugins()
        [{'name': 'my-plugin', 'version': '1.0.0', ...}]
        >>> manager.unload("my-plugin")
    """

    _instance: Optional["PluginManager"] = None

    def __init__(self):
        """Initialize plugin manager."""
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}

    @classmethod
    def get_instance(cls) -> "PluginManager":
        """Get the singleton plugin manager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, plugin_class: Type[Plugin]) -> Type[Plugin]:
        """Register a plugin class.

        Args:
            plugin_class: Plugin class to register

        Returns:
            The registered plugin class (for decorator use)

        Example:
            >>> @PluginManager.get_instance().register
            ... class MyPlugin(Plugin):
            ...     name = "my-plugin"
            ...     ...
        """
        # Create temporary instance to get name
        temp = plugin_class()
        self._plugin_classes[temp.name] = plugin_class
        return plugin_class

    def load(self, name: str) -> Plugin:
        """Load and initialize a plugin.

        Args:
            name: Plugin name or module path

        Returns:
            Loaded plugin instance

        Example:
            >>> plugin = manager.load("my-plugin")
            >>> plugin = manager.load("mypackage.myplugin:MyPlugin")
        """
        # Check if already loaded
        if name in self._plugins:
            return self._plugins[name]

        # Check registered plugins
        if name in self._plugin_classes:
            plugin = self._plugin_classes[name]()
        else:
            # Try to import as module
            plugin = self._import_plugin(name)

        # Check dependencies
        for dep in plugin.dependencies:
            if dep not in self._plugins:
                raise RuntimeError(
                    f"Plugin '{name}' requires '{dep}' which is not loaded"
                )

        # Initialize plugin
        plugin.setup()
        plugin._initialized = True
        self._plugins[plugin.name] = plugin

        # Register custom operators
        for op_name, op_fn in plugin.get_custom_ops().items():
            from .custom_op import register_custom_op
            try:
                register_custom_op(f"{plugin.name}.{op_name}", op_fn)
            except ValueError:
                pass  # Already registered

        return plugin

    def _is_allowed_module(self, module_path: str) -> bool:
        """Check if a module path is in the allowed list.

        Args:
            module_path: Module path to check

        Returns:
            True if allowed, False otherwise
        """
        for prefix in ALLOWED_PLUGIN_PREFIXES:
            if module_path.startswith(prefix):
                return True
        return False

    def _import_plugin(self, name: str) -> Plugin:
        """Import a plugin from a module path with security restrictions.

        Only allows importing plugins from whitelisted module prefixes
        to prevent arbitrary code execution.

        Args:
            name: Module path (e.g., "pyflame_myplugin:MyPlugin")

        Returns:
            Plugin instance

        Raises:
            SecurityError: If the module path is not in the allowed list
            ImportError: If the plugin cannot be imported
        """
        if ":" in name:
            module_path, class_name = name.rsplit(":", 1)
        else:
            module_path = name
            class_name = "Plugin"

        # Security check: Only allow importing from whitelisted namespaces
        if not self._is_allowed_module(module_path):
            logger.warning(
                f"Blocked plugin import from untrusted namespace: {module_path}"
            )
            raise SecurityError(
                f"Plugin '{name}' is not from a trusted namespace. "
                f"Allowed prefixes: {sorted(ALLOWED_PLUGIN_PREFIXES)}. "
                f"Set PYFLAME_ALLOWED_PLUGIN_PREFIXES environment variable to add custom prefixes."
            )

        try:
            logger.info(f"Importing plugin from {module_path}")
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)

            if not issubclass(plugin_class, Plugin):
                raise TypeError(f"{class_name} is not a Plugin subclass")

            return plugin_class()

        except ImportError as e:
            raise ImportError(f"Could not import plugin '{name}': {e}")
        except AttributeError:
            raise ImportError(f"Could not find '{class_name}' in '{module_path}'")

    def unload(self, name: str):
        """Unload a plugin.

        Args:
            name: Plugin name
        """
        if name not in self._plugins:
            return

        plugin = self._plugins[name]

        # Check for dependents
        for other_name, other_plugin in self._plugins.items():
            if name in other_plugin.dependencies:
                raise RuntimeError(
                    f"Cannot unload '{name}': required by '{other_name}'"
                )

        # Teardown plugin
        plugin.teardown()
        plugin._initialized = False

        del self._plugins[name]

    def get(self, name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins.

        Returns:
            List of plugin information dictionaries
        """
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "author": p.author,
                "tags": p.tags,
            }
            for p in self._plugins.values()
        ]

    def list_available(self) -> List[str]:
        """List all registered (available) plugins.

        Returns:
            List of plugin names
        """
        return list(self._plugin_classes.keys())


# Global plugin manager functions
def load_plugin(name: str) -> Plugin:
    """Load a plugin by name.

    Convenience function for PluginManager.load().

    Args:
        name: Plugin name or module path

    Returns:
        Loaded plugin instance

    Example:
        >>> plugin = load_plugin("pyflame-wandb")
        >>> plugin = load_plugin("mypackage.plugin:MyPlugin")
    """
    return PluginManager.get_instance().load(name)


def unload_plugin(name: str):
    """Unload a plugin.

    Convenience function for PluginManager.unload().

    Args:
        name: Plugin name
    """
    PluginManager.get_instance().unload(name)


def list_plugins() -> List[Dict[str, Any]]:
    """List all loaded plugins.

    Returns:
        List of plugin information dictionaries
    """
    return PluginManager.get_instance().list_plugins()


def register_plugin(plugin_class: Type[Plugin]) -> Type[Plugin]:
    """Register a plugin class.

    Decorator for registering plugin classes.

    Args:
        plugin_class: Plugin class to register

    Returns:
        The registered plugin class

    Example:
        >>> @register_plugin
        ... class MyPlugin(Plugin):
        ...     name = "my-plugin"
        ...     version = "1.0.0"
        ...
        ...     def setup(self):
        ...         print("Plugin loaded!")
        ...
        ...     def teardown(self):
        ...         print("Plugin unloaded!")
    """
    return PluginManager.get_instance().register(plugin_class)


def get_plugin(name: str) -> Optional[Plugin]:
    """Get a loaded plugin by name.

    Args:
        name: Plugin name

    Returns:
        Plugin instance or None
    """
    return PluginManager.get_instance().get(name)


# Entry points discovery (for installed packages)
def discover_plugins() -> List[str]:
    """Discover plugins from installed packages.

    Looks for entry points in the 'pyflame.plugins' group.

    Returns:
        List of discovered plugin names
    """
    discovered = []

    try:
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points
            eps = entry_points(group="pyflame.plugins")
        else:
            from importlib.metadata import entry_points
            eps = entry_points().get("pyflame.plugins", [])

        for ep in eps:
            discovered.append(ep.name)
            # Auto-register discovered plugins
            try:
                plugin_class = ep.load()
                PluginManager.get_instance().register(plugin_class)
            except Exception:
                pass

    except ImportError:
        pass

    return discovered
