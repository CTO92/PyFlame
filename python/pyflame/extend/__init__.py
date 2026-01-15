"""
PyFlame Extension Module.

Provides APIs for extending PyFlame with custom operators and plugins.
"""

from .custom_op import (
    CustomOp,
    custom_op,
    get_custom_op,
    list_custom_ops,
    register_custom_op,
)
from .plugin import (
    Plugin,
    PluginManager,
    list_plugins,
    load_plugin,
    register_plugin,
    unload_plugin,
)

__all__ = [
    # Custom operators
    "register_custom_op",
    "custom_op",
    "get_custom_op",
    "list_custom_ops",
    "CustomOp",
    # Plugins
    "Plugin",
    "PluginManager",
    "load_plugin",
    "unload_plugin",
    "list_plugins",
    "register_plugin",
]
