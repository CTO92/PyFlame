# Phase 4: Ecosystem - Architecture & Implementation Plan

**PyFlame Version:** Pre-Release Alpha 1.0
**Phase:** 4 of 4
**Focus:** Ecosystem Development
**Target Duration:** Months 18-24

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. Plans described here are subject to change based on community feedback and technical discoveries.

---

## Table of Contents

1. [Phase 4 Overview](#1-phase-4-overview)
2. [Developer Tools & IDE Integration](#2-developer-tools--ide-integration)
3. [Documentation System](#3-documentation-system)
4. [Distribution & Packaging](#4-distribution--packaging)
5. [Community Infrastructure](#5-community-infrastructure)
6. [Integrations & Interoperability](#6-integrations--interoperability)
7. [Production Deployment](#7-production-deployment)
8. [Benchmarking & Performance Analysis](#8-benchmarking--performance-analysis)
9. [Plugin & Extension System](#9-plugin--extension-system)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Technical Decisions](#11-technical-decisions)

---

## 1. Phase 4 Overview

### 1.1 Goals

Phase 4 transforms PyFlame from a functional framework into a complete ecosystem ready for production use and community adoption. The focus shifts from core functionality to developer experience, tooling, and integration.

| Component | Description |
|-----------|-------------|
| **Developer Tools** | IDE integrations, debuggers, profilers, visualization tools |
| **Documentation** | Auto-generated API docs, tutorials, migration guides, examples |
| **Distribution** | PyPI, conda, versioning, dependency management |
| **Community** | Contribution guidelines, RFC process, governance model |
| **Integrations** | ONNX, MLOps tools (MLflow, W&B), Jupyter, model converters |
| **Deployment** | Serving infrastructure, containerization, cloud deployment |
| **Benchmarking** | Performance suite, comparison tools, optimization guides |
| **Extensibility** | Plugin system, custom operator registration, extensions API |

### 1.2 Dependencies on Previous Phases

Phase 4 requires these components from previous phases:

- [x] Phase 1: Core tensor operations, CSL code generation, Python bindings
- [x] Phase 2: Autograd, neural network layers, optimizers, loss functions
- [x] Phase 3: Data loading, model serialization, pre-built models, training utilities, model hub

### 1.3 Key Design Principles

1. **Developer-First**: Every feature should improve developer productivity
2. **PyTorch Familiarity**: Maintain API compatibility where possible for easy migration
3. **Cerebras-Native**: Expose Cerebras-specific optimizations without hiding complexity
4. **Community-Driven**: Build infrastructure for sustainable open-source development
5. **Production-Ready**: Enable seamless transition from research to deployment

### 1.4 Success Metrics

| Metric | Target |
|--------|--------|
| Installation success rate | >95% across supported platforms |
| Documentation coverage | 100% public API documented |
| Time to first model training | <15 minutes for new users |
| PyTorch model conversion success | >80% of common architectures |
| Community engagement | Active contributors, issue response time <48h |

---

## 2. Developer Tools & IDE Integration

### 2.1 Overview

Modern ML development requires rich tooling support. Phase 4 will provide first-class IDE integration and debugging capabilities.

### 2.2 VSCode Extension

#### 2.2.1 Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Syntax Highlighting** | CSL syntax highlighting in embedded code | High |
| **IntelliSense** | Auto-completion for PyFlame APIs | High |
| **Tensor Shape Hints** | Display inferred tensor shapes inline | High |
| **Graph Visualization** | Interactive computation graph viewer | Medium |
| **Debugger Integration** | Step through PyFlame operations | Medium |
| **Profiler View** | Performance profiling results viewer | Medium |
| **CSL Preview** | Preview generated CSL code | Low |

#### 2.2.2 Extension Architecture

```
pyflame-vscode/
├── package.json                  # Extension manifest
├── src/
│   ├── extension.ts              # Main entry point
│   ├── providers/
│   │   ├── completionProvider.ts # IntelliSense
│   │   ├── hoverProvider.ts      # Tensor shape hints
│   │   ├── diagnosticsProvider.ts # Error detection
│   │   └── codeActionsProvider.ts # Quick fixes
│   ├── views/
│   │   ├── graphView.ts          # Graph visualization
│   │   ├── profilerView.ts       # Profiling results
│   │   └── tensorView.ts         # Tensor inspector
│   └── debugger/
│       ├── debugAdapter.ts       # Debug adapter protocol
│       └── pyflameDebugSession.ts
├── syntaxes/
│   └── csl.tmLanguage.json       # CSL syntax grammar
└── language-configuration.json
```

#### 2.2.3 Language Server Protocol

```python
# python/pyflame/tools/language_server.py

"""PyFlame Language Server Protocol implementation."""

import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TensorInfo:
    """Information about a tensor for IDE display."""
    name: str
    shape: List[int]
    dtype: str
    layout: str
    memory_bytes: int


@dataclass
class CompletionItem:
    """Auto-completion suggestion."""
    label: str
    kind: str  # 'function', 'class', 'module', 'variable'
    detail: str
    documentation: str
    insert_text: str


class PyFlameLanguageServer:
    """Language server for PyFlame IDE integration."""

    def __init__(self):
        self.workspace_root: Optional[str] = None
        self.tensor_cache: Dict[str, TensorInfo] = {}
        self.graph_cache: Dict[str, Any] = {}

    async def initialize(self, root_path: str) -> Dict[str, Any]:
        """Initialize the language server."""
        self.workspace_root = root_path
        return {
            "capabilities": {
                "completionProvider": {
                    "triggerCharacters": [".", "(", "["],
                    "resolveProvider": True,
                },
                "hoverProvider": True,
                "definitionProvider": True,
                "referencesProvider": True,
                "documentSymbolProvider": True,
                "codeActionProvider": True,
            }
        }

    async def get_completions(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> List[CompletionItem]:
        """Get auto-completion suggestions at cursor position."""
        # Analyze context and return relevant completions
        pass

    async def get_hover_info(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> Optional[str]:
        """Get hover information (tensor shapes, docs)."""
        # Return tensor shape and type information
        pass

    async def get_tensor_info(self, tensor_name: str) -> Optional[TensorInfo]:
        """Get detailed information about a tensor."""
        pass

    async def get_graph_visualization(
        self,
        output_tensor: str
    ) -> Dict[str, Any]:
        """Generate graph visualization data."""
        pass
```

### 2.3 PyFlame Debugger

#### 2.3.1 Debugging Features

```python
# python/pyflame/tools/debugger.py

"""PyFlame interactive debugger."""

from typing import Any, Callable, Dict, List, Optional
import pyflame as pf


class BreakpointType:
    """Types of breakpoints supported."""
    OPERATION = "operation"      # Break on specific op type
    TENSOR = "tensor"            # Break when tensor is created/modified
    SHAPE = "shape"              # Break on shape mismatch
    NAN = "nan"                  # Break on NaN/Inf detection
    MEMORY = "memory"            # Break on memory threshold
    CUSTOM = "custom"            # Custom condition


class PyFlameDebugger:
    """Interactive debugger for PyFlame computations."""

    def __init__(self):
        self.breakpoints: List[Breakpoint] = []
        self.watch_tensors: Dict[str, pf.Tensor] = {}
        self.history: List[DebugEvent] = []
        self._enabled = False

    def enable(self):
        """Enable debugging mode."""
        self._enabled = True
        pf.set_debug_mode(True)

    def disable(self):
        """Disable debugging mode."""
        self._enabled = False
        pf.set_debug_mode(False)

    def add_breakpoint(
        self,
        breakpoint_type: str,
        condition: Optional[Callable] = None,
        **kwargs
    ) -> int:
        """Add a breakpoint.

        Args:
            breakpoint_type: Type of breakpoint
            condition: Optional custom condition function
            **kwargs: Type-specific parameters

        Returns:
            Breakpoint ID
        """
        pass

    def remove_breakpoint(self, breakpoint_id: int):
        """Remove a breakpoint."""
        pass

    def watch(self, name: str, tensor: pf.Tensor):
        """Add tensor to watch list."""
        self.watch_tensors[name] = tensor

    def unwatch(self, name: str):
        """Remove tensor from watch list."""
        del self.watch_tensors[name]

    def inspect_tensor(self, tensor: pf.Tensor) -> Dict[str, Any]:
        """Get detailed tensor information.

        Returns:
            Dictionary with shape, dtype, stats, layout, memory info
        """
        return {
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "layout": str(tensor.layout),
            "numel": tensor.numel,
            "memory_bytes": tensor.numel * pf.dtype_size(tensor.dtype),
            "stats": {
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "has_nan": bool(tensor.isnan().any()),
                "has_inf": bool(tensor.isinf().any()),
            },
            "gradient": tensor.grad is not None,
        }

    def print_graph(self, tensor: pf.Tensor, depth: int = 3):
        """Print computation graph leading to tensor."""
        pass

    def step(self):
        """Execute next operation and pause."""
        pass

    def continue_execution(self):
        """Continue until next breakpoint."""
        pass


# Context manager for debugging
class debug_context:
    """Context manager for debugging a code block."""

    def __init__(self, break_on_nan: bool = True, break_on_error: bool = True):
        self.debugger = PyFlameDebugger()
        self.break_on_nan = break_on_nan
        self.break_on_error = break_on_error

    def __enter__(self):
        self.debugger.enable()
        if self.break_on_nan:
            self.debugger.add_breakpoint(BreakpointType.NAN)
        return self.debugger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.debugger.disable()
        return False


# Usage example:
# with pf.tools.debug_context() as dbg:
#     dbg.watch("output", model(x))
#     result = model(x)
```

### 2.4 Profiler

#### 2.4.1 Profiling Architecture

```python
# python/pyflame/tools/profiler.py

"""Performance profiling for PyFlame."""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import pyflame as pf


@dataclass
class ProfileEvent:
    """Single profiling event."""
    name: str
    category: str  # 'compute', 'memory', 'communication', 'host'
    start_time: float
    end_time: float
    memory_allocated: int = 0
    memory_freed: int = 0
    input_shapes: List[List[int]] = field(default_factory=list)
    output_shapes: List[List[int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class ProfileSummary:
    """Summary statistics from profiling."""
    total_time_ms: float
    compute_time_ms: float
    memory_time_ms: float
    communication_time_ms: float
    host_time_ms: float
    peak_memory_bytes: int
    total_operations: int
    operations_by_type: Dict[str, int]
    bottlenecks: List[str]


class Profiler:
    """PyFlame performance profiler."""

    def __init__(
        self,
        record_shapes: bool = True,
        record_memory: bool = True,
        with_stack: bool = False,
    ):
        self.record_shapes = record_shapes
        self.record_memory = record_memory
        self.with_stack = with_stack
        self.events: List[ProfileEvent] = []
        self._active = False

    def start(self):
        """Start profiling."""
        self._active = True
        self.events.clear()
        pf._set_profiling_enabled(True, self._event_callback)

    def stop(self) -> ProfileSummary:
        """Stop profiling and return summary."""
        self._active = False
        pf._set_profiling_enabled(False, None)
        return self._compute_summary()

    def _event_callback(self, event_data: Dict[str, Any]):
        """Callback for profiling events from C++ backend."""
        event = ProfileEvent(**event_data)
        self.events.append(event)

    def _compute_summary(self) -> ProfileSummary:
        """Compute profiling summary statistics."""
        if not self.events:
            return ProfileSummary(0, 0, 0, 0, 0, 0, 0, {}, [])

        total_time = sum(e.duration_ms for e in self.events)
        compute_time = sum(e.duration_ms for e in self.events
                         if e.category == 'compute')
        memory_time = sum(e.duration_ms for e in self.events
                         if e.category == 'memory')
        comm_time = sum(e.duration_ms for e in self.events
                       if e.category == 'communication')
        host_time = sum(e.duration_ms for e in self.events
                       if e.category == 'host')

        # Find peak memory
        current_memory = 0
        peak_memory = 0
        for event in self.events:
            current_memory += event.memory_allocated - event.memory_freed
            peak_memory = max(peak_memory, current_memory)

        # Count operations by type
        ops_by_type: Dict[str, int] = {}
        for event in self.events:
            if event.category == 'compute':
                ops_by_type[event.name] = ops_by_type.get(event.name, 0) + 1

        # Identify bottlenecks (ops taking >10% of total time)
        bottlenecks = []
        for event in self.events:
            if event.duration_ms > total_time * 0.1:
                bottlenecks.append(
                    f"{event.name}: {event.duration_ms:.2f}ms "
                    f"({event.duration_ms/total_time*100:.1f}%)"
                )

        return ProfileSummary(
            total_time_ms=total_time,
            compute_time_ms=compute_time,
            memory_time_ms=memory_time,
            communication_time_ms=comm_time,
            host_time_ms=host_time,
            peak_memory_bytes=peak_memory,
            total_operations=len(self.events),
            operations_by_type=ops_by_type,
            bottlenecks=bottlenecks[:10],  # Top 10 bottlenecks
        )

    def export_chrome_trace(self, path: str):
        """Export profiling data as Chrome trace format.

        Can be viewed at chrome://tracing
        """
        import json

        trace_events = []
        for event in self.events:
            trace_events.append({
                "name": event.name,
                "cat": event.category,
                "ph": "X",  # Complete event
                "ts": event.start_time * 1e6,  # Convert to microseconds
                "dur": (event.end_time - event.start_time) * 1e6,
                "pid": 1,
                "tid": 1,
                "args": event.metadata,
            })

        with open(path, 'w') as f:
            json.dump({"traceEvents": trace_events}, f)

    def export_tensorboard(self, log_dir: str):
        """Export profiling data for TensorBoard visualization."""
        pass

    def print_summary(self):
        """Print human-readable profiling summary."""
        summary = self._compute_summary()
        print("\n" + "=" * 60)
        print("PyFlame Profiling Summary")
        print("=" * 60)
        print(f"Total time: {summary.total_time_ms:.2f} ms")
        print(f"  - Compute: {summary.compute_time_ms:.2f} ms "
              f"({summary.compute_time_ms/summary.total_time_ms*100:.1f}%)")
        print(f"  - Memory: {summary.memory_time_ms:.2f} ms")
        print(f"  - Communication: {summary.communication_time_ms:.2f} ms")
        print(f"  - Host: {summary.host_time_ms:.2f} ms")
        print(f"Peak memory: {summary.peak_memory_bytes / 1e6:.2f} MB")
        print(f"Total operations: {summary.total_operations}")
        print("\nTop operations by count:")
        for op, count in sorted(summary.operations_by_type.items(),
                               key=lambda x: -x[1])[:10]:
            print(f"  {op}: {count}")
        if summary.bottlenecks:
            print("\nBottlenecks:")
            for b in summary.bottlenecks:
                print(f"  - {b}")
        print("=" * 60 + "\n")


@contextmanager
def profile(
    record_shapes: bool = True,
    record_memory: bool = True,
    print_summary: bool = True,
):
    """Context manager for profiling.

    Example:
        with pf.tools.profile() as prof:
            output = model(input)
        prof.export_chrome_trace("trace.json")
    """
    profiler = Profiler(record_shapes=record_shapes, record_memory=record_memory)
    profiler.start()
    try:
        yield profiler
    finally:
        profiler.stop()
        if print_summary:
            profiler.print_summary()
```

### 2.5 Graph Visualization

#### 2.5.1 Visualization Module

```python
# python/pyflame/tools/visualization.py

"""Computation graph visualization for PyFlame."""

from typing import Dict, List, Optional, Any
import pyflame as pf


class GraphVisualizer:
    """Visualize PyFlame computation graphs."""

    def __init__(self, graph: pf.Graph):
        self.graph = graph

    def to_dot(self, show_shapes: bool = True, show_dtypes: bool = True) -> str:
        """Convert graph to DOT format for Graphviz.

        Args:
            show_shapes: Include tensor shapes in labels
            show_dtypes: Include data types in labels

        Returns:
            DOT format string
        """
        lines = ["digraph PyFlameGraph {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box, style=filled];")

        # Add nodes
        for node in self.graph.nodes:
            label = node.op_type
            if show_shapes:
                label += f"\\n{node.output_shape}"
            if show_dtypes:
                label += f"\\n{node.output_dtype}"

            # Color by operation type
            color = self._get_node_color(node.op_type)
            lines.append(f'  {node.id} [label="{label}", fillcolor="{color}"];')

        # Add edges
        for node in self.graph.nodes:
            for input_node in node.inputs:
                lines.append(f"  {input_node.id} -> {node.id};")

        lines.append("}")
        return "\n".join(lines)

    def _get_node_color(self, op_type: str) -> str:
        """Get color for node based on operation type."""
        colors = {
            "matmul": "#ff9999",      # Red for compute-heavy
            "conv2d": "#ff9999",
            "linear": "#ff9999",
            "relu": "#99ff99",        # Green for activation
            "sigmoid": "#99ff99",
            "softmax": "#99ff99",
            "add": "#9999ff",         # Blue for elementwise
            "mul": "#9999ff",
            "sum": "#ffff99",         # Yellow for reduction
            "mean": "#ffff99",
            "input": "#ffffff",       # White for inputs
            "output": "#cccccc",      # Gray for outputs
        }
        return colors.get(op_type.lower(), "#ffccff")

    def to_html(self, interactive: bool = True) -> str:
        """Generate HTML visualization with D3.js.

        Args:
            interactive: Enable interactive features (zoom, pan, tooltips)

        Returns:
            HTML string
        """
        pass

    def save_png(self, path: str, dpi: int = 150):
        """Save graph visualization as PNG image.

        Requires graphviz to be installed.
        """
        import subprocess

        dot_content = self.to_dot()

        # Use graphviz dot command
        process = subprocess.run(
            ["dot", "-Tpng", f"-Gdpi={dpi}"],
            input=dot_content.encode(),
            capture_output=True,
        )

        if process.returncode != 0:
            raise RuntimeError(f"Graphviz error: {process.stderr.decode()}")

        with open(path, 'wb') as f:
            f.write(process.stdout)

    def save_svg(self, path: str):
        """Save graph visualization as SVG."""
        import subprocess

        dot_content = self.to_dot()
        process = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot_content.encode(),
            capture_output=True,
        )

        if process.returncode != 0:
            raise RuntimeError(f"Graphviz error: {process.stderr.decode()}")

        with open(path, 'wb') as f:
            f.write(process.stdout)

    def display_jupyter(self):
        """Display graph in Jupyter notebook."""
        try:
            from IPython.display import SVG, display
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
                self.save_svg(f.name)
                display(SVG(filename=f.name))
        except ImportError:
            print("IPython not available. Use save_png() or save_svg() instead.")


def visualize_graph(tensor: pf.Tensor, **kwargs) -> GraphVisualizer:
    """Convenience function to visualize tensor's computation graph.

    Example:
        output = model(input)
        pf.tools.visualize_graph(output).save_png("model_graph.png")
    """
    graph = pf.get_graph(tensor)
    return GraphVisualizer(graph, **kwargs)
```

---

## 3. Documentation System

### 3.1 Overview

A comprehensive documentation system is essential for adoption. Phase 4 will implement auto-generated API docs, interactive tutorials, and migration resources.

### 3.2 Documentation Architecture

```
docs/
├── source/
│   ├── conf.py                    # Sphinx configuration
│   ├── index.rst                  # Landing page
│   ├── getting_started/
│   │   ├── installation.rst
│   │   ├── quickstart.rst
│   │   └── first_model.rst
│   ├── tutorials/
│   │   ├── index.rst
│   │   ├── basics/
│   │   │   ├── tensors.ipynb
│   │   │   ├── operations.ipynb
│   │   │   └── autograd.ipynb
│   │   ├── intermediate/
│   │   │   ├── custom_layers.ipynb
│   │   │   ├── data_loading.ipynb
│   │   │   └── training_loop.ipynb
│   │   └── advanced/
│   │       ├── mesh_layouts.ipynb
│   │       ├── csl_integration.ipynb
│   │       └── optimization.ipynb
│   ├── api/
│   │   ├── index.rst
│   │   ├── tensor.rst
│   │   ├── nn.rst
│   │   ├── optim.rst
│   │   └── data.rst
│   ├── migration/
│   │   ├── from_pytorch.rst
│   │   └── from_tensorflow.rst
│   ├── deployment/
│   │   ├── serving.rst
│   │   ├── optimization.rst
│   │   └── production.rst
│   └── contributing/
│       ├── development.rst
│       ├── style_guide.rst
│       └── testing.rst
├── _templates/                    # Custom Sphinx templates
├── _static/                       # Static assets
│   ├── css/
│   ├── js/
│   └── images/
└── build/                         # Generated documentation
```

### 3.3 Sphinx Configuration

```python
# docs/source/conf.py

"""Sphinx configuration for PyFlame documentation."""

import os
import sys

# Add pyflame to path for autodoc
sys.path.insert(0, os.path.abspath('../../python'))

# Project information
project = 'PyFlame'
copyright = '2026, PyFlame Contributors'
author = 'PyFlame Contributors'
version = '1.0'
release = '1.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',           # Auto-generate from docstrings
    'sphinx.ext.napoleon',          # Google/NumPy style docstrings
    'sphinx.ext.intersphinx',       # Link to other projects
    'sphinx.ext.viewcode',          # Source code links
    'sphinx.ext.mathjax',           # Math rendering
    'sphinx_autodoc_typehints',     # Type hints in docs
    'nbsphinx',                     # Jupyter notebook support
    'sphinx_copybutton',            # Copy button for code blocks
    'sphinx_tabs.tabs',             # Tabbed content
    'sphinxcontrib.mermaid',        # Diagrams
]

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Napoleon configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = {
    'Tensor': 'pyflame.Tensor',
    'Module': 'pyflame.nn.Module',
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Theme configuration
html_theme = 'furo'
html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#FF6B35',
        'color-brand-content': '#FF6B35',
    },
    'dark_css_variables': {
        'color-brand-primary': '#FF8C61',
        'color-brand-content': '#FF8C61',
    },
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
}

html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/images/pyflame-logo.svg'
html_favicon = '_static/images/favicon.ico'

# NBSphinx configuration
nbsphinx_execute = 'auto'
nbsphinx_kernel_name = 'python3'
nbsphinx_timeout = 300

# Copy button configuration
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True
```

### 3.4 Migration Guide from PyTorch

```python
# docs/source/migration/pytorch_converter.py

"""Tools for migrating from PyTorch to PyFlame."""

from typing import Dict, Any, List, Optional
import ast
import re


# API mapping from PyTorch to PyFlame
PYTORCH_TO_PYFLAME_MAPPING = {
    # Imports
    "import torch": "import pyflame as pf",
    "from torch": "from pyflame",
    "torch.Tensor": "pf.Tensor",
    "torch.nn": "pf.nn",
    "torch.optim": "pf.optim",

    # Tensor creation
    "torch.zeros": "pf.zeros",
    "torch.ones": "pf.ones",
    "torch.randn": "pf.randn",
    "torch.rand": "pf.rand",
    "torch.tensor": "pf.tensor",
    "torch.from_numpy": "pf.from_numpy",
    "torch.arange": "pf.arange",
    "torch.linspace": "pf.linspace",
    "torch.empty": "pf.empty",
    "torch.full": "pf.full",

    # Operations
    "torch.matmul": "pf.matmul",
    "torch.mm": "pf.matmul",
    "torch.bmm": "pf.bmm",
    "torch.relu": "pf.relu",
    "torch.sigmoid": "pf.sigmoid",
    "torch.tanh": "pf.tanh",
    "torch.softmax": "pf.softmax",
    "torch.log_softmax": "pf.log_softmax",
    "torch.cat": "pf.cat",
    "torch.stack": "pf.stack",
    "torch.split": "pf.split",
    "torch.chunk": "pf.chunk",

    # Reductions
    "torch.sum": "pf.sum",
    "torch.mean": "pf.mean",
    "torch.max": "pf.max",
    "torch.min": "pf.min",
    "torch.argmax": "pf.argmax",
    "torch.argmin": "pf.argmin",

    # Data types
    "torch.float32": "pf.float32",
    "torch.float16": "pf.float16",
    "torch.bfloat16": "pf.bfloat16",
    "torch.int32": "pf.int32",
    "torch.int64": "pf.int64",

    # Autograd
    "torch.no_grad": "pf.no_grad",
    "torch.enable_grad": "pf.enable_grad",
    ".backward()": ".backward()",
    ".grad": ".grad",

    # Neural network layers
    "nn.Linear": "nn.Linear",
    "nn.Conv2d": "nn.Conv2d",
    "nn.Conv1d": "nn.Conv1d",
    "nn.BatchNorm2d": "nn.BatchNorm2d",
    "nn.LayerNorm": "nn.LayerNorm",
    "nn.Dropout": "nn.Dropout",
    "nn.ReLU": "nn.ReLU",
    "nn.GELU": "nn.GELU",
    "nn.Softmax": "nn.Softmax",
    "nn.Sequential": "nn.Sequential",
    "nn.ModuleList": "nn.ModuleList",
    "nn.Embedding": "nn.Embedding",
    "nn.MultiheadAttention": "nn.MultiheadAttention",

    # Loss functions
    "nn.CrossEntropyLoss": "nn.CrossEntropyLoss",
    "nn.MSELoss": "nn.MSELoss",
    "nn.BCELoss": "nn.BCELoss",
    "nn.BCEWithLogitsLoss": "nn.BCEWithLogitsLoss",
    "nn.L1Loss": "nn.L1Loss",

    # Optimizers
    "optim.SGD": "optim.SGD",
    "optim.Adam": "optim.Adam",
    "optim.AdamW": "optim.AdamW",
    "optim.RMSprop": "optim.RMSprop",

    # Learning rate schedulers
    "optim.lr_scheduler.StepLR": "optim.StepLR",
    "optim.lr_scheduler.CosineAnnealingLR": "optim.CosineAnnealingLR",
    "optim.lr_scheduler.ReduceLROnPlateau": "optim.ReduceLROnPlateau",
}

# Features that require manual intervention
MANUAL_MIGRATION_REQUIRED = {
    ".cuda()": "PyFlame uses MeshLayout instead of device placement",
    ".to(device)": "Use pf.Tensor with layout parameter instead",
    "torch.device": "PyFlame uses MeshLayout for hardware placement",
    "DataParallel": "Use PyFlame's distributed training API",
    "DistributedDataParallel": "Use PyFlame's distributed training API",
    "torch.jit": "PyFlame uses static compilation; see CSL backend docs",
    "torch.compile": "PyFlame compiles automatically for Cerebras",
}


class PyTorchToPyFlameConverter:
    """Convert PyTorch code to PyFlame."""

    def __init__(self):
        self.warnings: List[str] = []
        self.manual_interventions: List[Dict[str, Any]] = []

    def convert_file(self, pytorch_code: str) -> str:
        """Convert PyTorch Python code to PyFlame.

        Args:
            pytorch_code: String containing PyTorch code

        Returns:
            Converted PyFlame code
        """
        self.warnings.clear()
        self.manual_interventions.clear()

        result = pytorch_code

        # Apply direct mappings
        for pytorch, pyflame in PYTORCH_TO_PYFLAME_MAPPING.items():
            result = result.replace(pytorch, pyflame)

        # Check for features requiring manual intervention
        for pattern, message in MANUAL_MIGRATION_REQUIRED.items():
            if pattern in result:
                self.manual_interventions.append({
                    "pattern": pattern,
                    "message": message,
                    "line_numbers": self._find_line_numbers(result, pattern),
                })

        return result

    def _find_line_numbers(self, code: str, pattern: str) -> List[int]:
        """Find line numbers containing pattern."""
        lines = code.split('\n')
        return [i + 1 for i, line in enumerate(lines) if pattern in line]

    def get_migration_report(self) -> str:
        """Generate a migration report."""
        report = ["# PyTorch to PyFlame Migration Report\n"]

        if not self.manual_interventions:
            report.append("No manual interventions required.\n")
        else:
            report.append("## Manual Interventions Required\n")
            for item in self.manual_interventions:
                report.append(f"### `{item['pattern']}`\n")
                report.append(f"Lines: {item['line_numbers']}\n")
                report.append(f"Action: {item['message']}\n\n")

        return "\n".join(report)


def convert_pytorch_model(model_code: str) -> str:
    """Convenience function to convert PyTorch model code.

    Example:
        pytorch_code = '''
        import torch
        import torch.nn as nn

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(784, 256)
                self.relu = nn.ReLU()
                self.output = nn.Linear(256, 10)

            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                return self.output(x)
        '''

        pyflame_code = convert_pytorch_model(pytorch_code)
    """
    converter = PyTorchToPyFlameConverter()
    result = converter.convert_file(model_code)

    if converter.manual_interventions:
        print("Warning: Some features require manual migration:")
        print(converter.get_migration_report())

    return result
```

---

## 4. Distribution & Packaging

### 4.1 Overview

Phase 4 will establish robust distribution channels for PyFlame, enabling easy installation across different platforms and environments.

### 4.2 Package Structure

```
pyflame/
├── pyproject.toml               # Modern Python packaging
├── setup.py                     # Legacy compatibility
├── setup.cfg                    # Setup configuration
├── MANIFEST.in                  # Include non-Python files
├── LICENSE                      # Apache 2.0
├── README.md                    # Project readme
├── python/
│   └── pyflame/                 # Python package
│       ├── __init__.py
│       ├── _version.py          # Version management
│       └── ...
├── include/                     # C++ headers
├── src/                         # C++ source
├── cmake/                       # CMake modules
├── scripts/
│   ├── build_wheels.py          # Wheel building script
│   └── run_tests.py             # Test runner
└── .github/
    └── workflows/
        ├── ci.yml               # CI/CD pipeline
        ├── release.yml          # Release automation
        └── docs.yml             # Documentation deployment
```

### 4.3 pyproject.toml Configuration

```toml
# pyproject.toml

[build-system]
requires = [
    "scikit-build-core>=0.5",
    "pybind11>=2.11",
    "cmake>=3.18",
    "ninja",
]
build-backend = "scikit_build_core.build"

[project]
name = "pyflame"
version = "1.0.0"
description = "Native Deep Learning Framework for Cerebras WSE"
readme = "README.md"
license = {text = "Apache-2.0"}
authors = [
    {name = "PyFlame Contributors", email = "pyflame@example.com"}
]
maintainers = [
    {name = "PyFlame Team", email = "pyflame@example.com"}
]
keywords = [
    "deep-learning",
    "machine-learning",
    "cerebras",
    "tensor",
    "neural-network",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: C++",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.0",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
    "pre-commit>=3.0",
]
docs = [
    "sphinx>=6.0",
    "furo>=2023.0",
    "sphinx-autodoc-typehints>=1.23",
    "nbsphinx>=0.9",
    "sphinx-copybutton>=0.5",
    "sphinx-tabs>=3.4",
]
tools = [
    "graphviz>=0.20",
    "matplotlib>=3.5",
    "tensorboard>=2.12",
]
all = [
    "pyflame[dev,docs,tools]",
]

[project.urls]
Homepage = "https://pyflame.dev"
Documentation = "https://docs.pyflame.dev"
Repository = "https://github.com/pyflame/pyflame"
Changelog = "https://github.com/pyflame/pyflame/blob/main/CHANGELOG.md"
Issues = "https://github.com/pyflame/pyflame/issues"

[project.scripts]
pyflame = "pyflame.cli:main"

[tool.scikit-build]
cmake.minimum-version = "3.18"
cmake.build-type = "Release"
wheel.packages = ["python/pyflame"]
wheel.py-api = "cp312"
sdist.include = [
    "include/**/*.hpp",
    "src/**/*.cpp",
    "cmake/**/*.cmake",
]

[tool.scikit-build.cmake.define]
BUILD_PYTHON_BINDINGS = "ON"
BUILD_TESTS = "OFF"
BUILD_EXAMPLES = "OFF"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311", "py312"]
include = '\.pyi?$'

[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "UP", "B"]
ignore = ["E501"]
target-version = "py38"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

### 4.4 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml

name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install linters
        run: pip install black ruff mypy
      - name: Run black
        run: black --check python/
      - name: Run ruff
        run: ruff check python/
      - name: Run mypy
        run: mypy python/pyflame/

  build:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies (Ubuntu)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y ninja-build

      - name: Install dependencies (macOS)
        if: runner.os == 'macOS'
        run: brew install ninja

      - name: Install dependencies (Windows)
        if: runner.os == 'Windows'
        run: choco install ninja

      - name: Build and install
        run: pip install -e ".[dev]"

      - name: Run tests
        run: pytest tests/ -v --cov=pyflame --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: coverage.xml

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -e ".[docs]"
      - name: Build docs
        run: |
          cd docs
          sphinx-build -b html source build/html
      - name: Upload docs artifact
        uses: actions/upload-artifact@v3
        with:
          name: documentation
          path: docs/build/html/
```

### 4.5 Release Workflow

```yaml
# .github/workflows/release.yml

name: Release

on:
  release:
    types: [published]

jobs:
  build-wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Build wheels
        uses: pypa/cibuildwheel@v2.16
        env:
          CIBW_ARCHS_LINUX: x86_64
          CIBW_ARCHS_MACOS: x86_64 arm64
          CIBW_ARCHS_WINDOWS: AMD64
          CIBW_SKIP: "*-musllinux* pp*"
          CIBW_BUILD: "cp38-* cp39-* cp310-* cp311-* cp312-*"

      - uses: actions/upload-artifact@v3
        with:
          name: wheels
          path: ./wheelhouse/*.whl

  build-sdist:
    name: Build source distribution
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Build sdist
        run: |
          pip install build
          python -m build --sdist
      - uses: actions/upload-artifact@v3
        with:
          name: sdist
          path: dist/*.tar.gz

  publish:
    needs: [build-wheels, build-sdist]
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write

    steps:
      - uses: actions/download-artifact@v3
        with:
          name: wheels
          path: dist/
      - uses: actions/download-artifact@v3
        with:
          name: sdist
          path: dist/
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

---

## 5. Community Infrastructure

### 5.1 Contribution Guidelines

```markdown
# Contributing to PyFlame

We welcome contributions to PyFlame! This document provides guidelines
for contributing to the project.

## Code of Conduct

PyFlame has adopted a Code of Conduct that we expect project participants
to adhere to. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

1. Check if the bug is already reported in [Issues](https://github.com/pyflame/pyflame/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - PyFlame version, Python version, OS
   - Minimal code example if possible

### Suggesting Features

1. Check existing [Discussions](https://github.com/pyflame/pyflame/discussions) and Issues
2. Create a new Discussion in the "Ideas" category
3. Include:
   - Problem you're trying to solve
   - Proposed solution
   - Alternatives considered
   - Impact on existing users

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest tests/`
6. Format code: `black python/` and `ruff check python/`
7. Commit with clear message: `git commit -m "Add feature X"`
8. Push to your fork: `git push origin feature/your-feature`
9. Create Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/pyflame/pyflame.git
cd pyflame

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use Black for formatting (line length 88)
- Use type hints for all public functions
- Write docstrings in Google style

### C++ Style

- Follow Google C++ Style Guide
- Use clang-format for formatting
- Use modern C++ (C++17 minimum)

### Documentation

- All public APIs must have docstrings
- Include examples in docstrings
- Update relevant documentation for changes

### Testing

- Write tests for all new functionality
- Maintain >90% code coverage
- Tests should be fast and deterministic
```

### 5.2 RFC Process

```markdown
# PyFlame RFC Process

Significant changes to PyFlame require a Request for Comments (RFC).

## When is an RFC Required?

- New API additions or changes to existing APIs
- Changes to the compilation pipeline
- New major features
- Breaking changes
- Changes to project governance

## RFC Template

```markdown
# RFC: [Title]

## Summary

One paragraph explanation of the proposal.

## Motivation

Why is this change necessary? What problem does it solve?

## Detailed Design

Technical details of the implementation:
- API changes
- Implementation approach
- Performance considerations
- Backward compatibility

## Alternatives Considered

What other approaches were considered and why were they rejected?

## Migration Path

How will existing users migrate to this change?

## Unresolved Questions

What aspects of the design are still uncertain?
```

## RFC Lifecycle

1. **Draft**: Initial proposal, open for feedback
2. **Discussion**: Community review period (minimum 2 weeks)
3. **Final Comment Period**: Last call for feedback (1 week)
4. **Accepted/Rejected**: Decision by maintainers
5. **Implemented**: RFC is being implemented
6. **Complete**: Implementation merged

## Submitting an RFC

1. Fork the repository
2. Copy `rfcs/0000-template.md` to `rfcs/0000-my-feature.md`
3. Fill in the RFC
4. Submit a pull request
5. Announce on Discussions
```

---

## 6. Integrations & Interoperability

### 6.1 ONNX Import/Export

#### 6.1.1 Architecture

```python
# python/pyflame/onnx/__init__.py

"""ONNX import/export for PyFlame models."""

from .export import export_onnx, export_onnx_from_traced
from .import_ import import_onnx, ONNXModel
from .converter import ONNXToPyFlameConverter, PyFlameToONNXConverter


__all__ = [
    "export_onnx",
    "export_onnx_from_traced",
    "import_onnx",
    "ONNXModel",
    "ONNXToPyFlameConverter",
    "PyFlameToONNXConverter",
]
```

#### 6.1.2 Export Implementation

```python
# python/pyflame/onnx/export.py

"""Export PyFlame models to ONNX format."""

from typing import Dict, List, Optional, Tuple, Union, Any
import pyflame as pf
import pyflame.nn as nn


def export_onnx(
    model: nn.Module,
    args: Union[pf.Tensor, Tuple[pf.Tensor, ...]],
    path: str,
    *,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
    do_constant_folding: bool = True,
    verbose: bool = False,
) -> None:
    """Export PyFlame model to ONNX format.

    Args:
        model: PyFlame model to export
        args: Example input(s) for tracing
        path: Output ONNX file path
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Specify dynamic dimensions
        opset_version: ONNX opset version
        do_constant_folding: Optimize constant operations
        verbose: Print export information

    Example:
        >>> model = MyModel()
        >>> x = pf.randn([1, 3, 224, 224])
        >>> pf.onnx.export_onnx(
        ...     model, x, "model.onnx",
        ...     input_names=["input"],
        ...     output_names=["output"],
        ...     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
        ... )
    """
    import onnx

    # Trace the model
    model.eval()
    with pf.no_grad():
        traced = pf.trace(model, args)

    # Get graph
    graph = traced.graph

    # Convert to ONNX
    converter = PyFlameToONNXConverter(
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
    )

    onnx_model = converter.convert(
        graph,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Validate
    onnx.checker.check_model(onnx_model)

    # Save
    onnx.save(onnx_model, path)

    if verbose:
        print(f"Exported ONNX model to {path}")
        print(f"  Opset version: {opset_version}")
        print(f"  Inputs: {input_names or 'auto'}")
        print(f"  Outputs: {output_names or 'auto'}")


class PyFlameToONNXConverter:
    """Convert PyFlame graph to ONNX model."""

    # Mapping from PyFlame ops to ONNX ops
    OP_MAPPING = {
        "add": "Add",
        "sub": "Sub",
        "mul": "Mul",
        "div": "Div",
        "matmul": "MatMul",
        "relu": "Relu",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "softmax": "Softmax",
        "gelu": "Gelu",
        "sum": "ReduceSum",
        "mean": "ReduceMean",
        "max": "ReduceMax",
        "min": "ReduceMin",
        "transpose": "Transpose",
        "reshape": "Reshape",
        "concat": "Concat",
        "split": "Split",
        "conv2d": "Conv",
        "maxpool2d": "MaxPool",
        "avgpool2d": "AveragePool",
        "batchnorm2d": "BatchNormalization",
        "layernorm": "LayerNormalization",
        "dropout": "Dropout",
        "linear": "Gemm",
    }

    def __init__(self, opset_version: int = 17, do_constant_folding: bool = True):
        self.opset_version = opset_version
        self.do_constant_folding = do_constant_folding

    def convert(
        self,
        graph: pf.Graph,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> Any:
        """Convert PyFlame graph to ONNX model."""
        import onnx
        from onnx import helper, TensorProto

        nodes = []
        inputs = []
        outputs = []
        initializers = []

        # Process graph nodes
        for node in graph.nodes:
            onnx_node = self._convert_node(node)
            if onnx_node is not None:
                nodes.append(onnx_node)

        # Create ONNX graph
        onnx_graph = helper.make_graph(
            nodes=nodes,
            name="pyflame_model",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )

        # Create ONNX model
        onnx_model = helper.make_model(
            onnx_graph,
            opset_imports=[helper.make_opsetid("", self.opset_version)],
        )

        return onnx_model

    def _convert_node(self, node: Any) -> Any:
        """Convert single PyFlame node to ONNX node."""
        pass
```

#### 6.1.3 Import Implementation

```python
# python/pyflame/onnx/import_.py

"""Import ONNX models into PyFlame."""

from typing import Dict, List, Optional, Any
import pyflame as pf
import pyflame.nn as nn


class ONNXModel(nn.Module):
    """PyFlame module wrapping an imported ONNX model."""

    def __init__(self, onnx_path: str):
        super().__init__()
        self._onnx_path = onnx_path
        self._graph = None
        self._weights = {}
        self._load_model()

    def _load_model(self):
        """Load and convert ONNX model."""
        import onnx

        onnx_model = onnx.load(self._onnx_path)
        onnx.checker.check_model(onnx_model)

        converter = ONNXToPyFlameConverter()
        self._graph, self._weights = converter.convert(onnx_model)

        # Register weights as parameters
        for name, weight in self._weights.items():
            self.register_parameter(name, pf.Parameter(weight))

    def forward(self, *args):
        """Execute the ONNX model."""
        return pf.execute_graph(self._graph, args, self._weights)


def import_onnx(path: str) -> ONNXModel:
    """Import ONNX model as PyFlame module.

    Args:
        path: Path to ONNX model file

    Returns:
        PyFlame module that can be used like any other module

    Example:
        >>> model = pf.onnx.import_onnx("model.onnx")
        >>> output = model(input_tensor)
    """
    return ONNXModel(path)


class ONNXToPyFlameConverter:
    """Convert ONNX model to PyFlame graph."""

    # Mapping from ONNX ops to PyFlame ops
    OP_MAPPING = {
        "Add": "add",
        "Sub": "sub",
        "Mul": "mul",
        "Div": "div",
        "MatMul": "matmul",
        "Gemm": "linear",
        "Relu": "relu",
        "Sigmoid": "sigmoid",
        "Tanh": "tanh",
        "Softmax": "softmax",
        "Gelu": "gelu",
        "ReduceSum": "sum",
        "ReduceMean": "mean",
        "ReduceMax": "max",
        "ReduceMin": "min",
        "Transpose": "transpose",
        "Reshape": "reshape",
        "Concat": "concat",
        "Split": "split",
        "Conv": "conv2d",
        "MaxPool": "maxpool2d",
        "AveragePool": "avgpool2d",
        "BatchNormalization": "batchnorm2d",
        "LayerNormalization": "layernorm",
        "Dropout": "dropout",
    }

    def convert(self, onnx_model: Any) -> tuple:
        """Convert ONNX model to PyFlame graph and weights."""
        pass
```

### 6.2 MLOps Integrations

#### 6.2.1 Weights & Biases Integration

```python
# python/pyflame/integrations/wandb.py

"""Weights & Biases integration for PyFlame."""

from typing import Any, Dict, Optional
import pyflame as pf
from pyflame.training.callbacks import Callback


class WandBCallback(Callback):
    """Callback for logging to Weights & Biases.

    Example:
        >>> import wandb
        >>> wandb.init(project="my-project")
        >>> trainer = Trainer(
        ...     model=model,
        ...     callbacks=[WandBCallback()]
        ... )
    """

    def __init__(
        self,
        log_model: bool = False,
        log_gradients: bool = False,
        log_freq: int = 100,
    ):
        self.log_model = log_model
        self.log_gradients = log_gradients
        self.log_freq = log_freq
        self._step = 0

    def on_train_begin(self, trainer, **kwargs):
        """Log model architecture."""
        import wandb

        if self.log_model:
            wandb.watch(
                trainer.model,
                log="all" if self.log_gradients else "parameters",
                log_freq=self.log_freq,
            )

    def on_batch_end(self, batch, logs=None, **kwargs):
        """Log batch metrics."""
        import wandb

        self._step += 1
        if logs:
            wandb.log(logs, step=self._step)

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        """Log epoch metrics."""
        import wandb

        if logs:
            epoch_logs = {f"epoch/{k}": v for k, v in logs.items()}
            wandb.log(epoch_logs, step=self._step)

    def on_train_end(self, trainer, **kwargs):
        """Save final model artifact."""
        import wandb

        if self.log_model:
            artifact = wandb.Artifact(
                "model",
                type="model",
                metadata={"framework": "pyflame"},
            )

            # Save model weights
            model_path = "model.pf"
            pf.save(trainer.model.state_dict(), model_path)
            artifact.add_file(model_path)

            wandb.log_artifact(artifact)
```

#### 6.2.2 MLflow Integration

```python
# python/pyflame/integrations/mlflow.py

"""MLflow integration for PyFlame."""

from typing import Any, Dict, Optional
import pyflame as pf
from pyflame.training.callbacks import Callback


class MLflowCallback(Callback):
    """Callback for logging to MLflow.

    Example:
        >>> import mlflow
        >>> mlflow.set_experiment("my-experiment")
        >>> trainer = Trainer(
        ...     model=model,
        ...     callbacks=[MLflowCallback()]
        ... )
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        log_models: bool = True,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_models = log_models
        self._run = None

    def on_train_begin(self, trainer, **kwargs):
        """Start MLflow run."""
        import mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        self._run = mlflow.start_run(run_name=self.run_name)

        # Log hyperparameters
        if hasattr(trainer, 'config'):
            mlflow.log_params(trainer.config.__dict__)

    def on_batch_end(self, batch, logs=None, **kwargs):
        """Log batch metrics."""
        import mlflow

        if logs:
            for key, value in logs.items():
                mlflow.log_metric(key, value, step=batch)

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        """Log epoch metrics."""
        import mlflow

        if logs:
            for key, value in logs.items():
                mlflow.log_metric(f"epoch_{key}", value, step=epoch)

    def on_train_end(self, trainer, **kwargs):
        """Save model and end run."""
        import mlflow

        if self.log_models:
            # Save model
            model_path = "model"
            pf.save(trainer.model.state_dict(), f"{model_path}.pf")
            mlflow.log_artifact(f"{model_path}.pf")

        mlflow.end_run()
```

### 6.3 Jupyter Integration

```python
# python/pyflame/integrations/jupyter.py

"""Jupyter notebook integration for PyFlame."""

from typing import Any, Optional
import pyflame as pf


def setup_jupyter():
    """Setup PyFlame for Jupyter notebooks.

    Call this at the start of a notebook to enable:
    - Rich tensor display
    - Graph visualization
    - Progress bars

    Example:
        >>> import pyflame as pf
        >>> pf.setup_jupyter()
    """
    try:
        from IPython import get_ipython
        from IPython.display import display, HTML

        ip = get_ipython()
        if ip is None:
            return

        # Register tensor formatter
        formatter = ip.display_formatter.formatters['text/html']
        formatter.for_type(pf.Tensor, _tensor_to_html)

        # Register graph formatter
        formatter.for_type(pf.Graph, _graph_to_html)

        print("PyFlame Jupyter integration enabled")

    except ImportError:
        pass


def _tensor_to_html(tensor: pf.Tensor) -> str:
    """Convert tensor to HTML for Jupyter display."""
    import numpy as np

    shape_str = "x".join(str(s) for s in tensor.shape)
    dtype_str = str(tensor.dtype)

    html = f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; font-family: monospace;">
        <b>PyFlame Tensor</b><br>
        <b>Shape:</b> [{shape_str}]<br>
        <b>DType:</b> {dtype_str}<br>
        <b>Layout:</b> {tensor.layout}<br>
    """

    # Show data preview for small tensors
    if tensor.numel <= 100:
        data = tensor.numpy()
        if data.ndim <= 2:
            html += f"<b>Data:</b><br><pre>{np.array2string(data, precision=4)}</pre>"
    else:
        html += f"<b>Elements:</b> {tensor.numel:,}<br>"
        stats = {
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "mean": float(tensor.mean()),
        }
        html += f"<b>Stats:</b> min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}"

    html += "</div>"
    return html


def _graph_to_html(graph: pf.Graph) -> str:
    """Convert computation graph to HTML visualization."""
    from pyflame.tools.visualization import GraphVisualizer

    viz = GraphVisualizer(graph)
    svg = viz.to_svg_string()

    return f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
        <b>PyFlame Computation Graph</b><br>
        <b>Nodes:</b> {graph.num_nodes}<br>
        <div style="max-height: 500px; overflow: auto;">
            {svg}
        </div>
    </div>
    """
```

---

## 7. Production Deployment

### 7.1 Overview

Phase 4 provides infrastructure for deploying PyFlame models in production environments.

### 7.2 Model Serving

#### 7.2.1 Serving Architecture

```python
# python/pyflame/serving/__init__.py

"""Model serving infrastructure for PyFlame."""

from .server import ModelServer, serve
from .client import ModelClient
from .inference import InferenceEngine, optimize_for_inference


__all__ = [
    "ModelServer",
    "serve",
    "ModelClient",
    "InferenceEngine",
    "optimize_for_inference",
]
```

#### 7.2.2 Inference Engine

```python
# python/pyflame/serving/inference.py

"""Optimized inference engine for PyFlame models."""

from typing import Any, Dict, List, Optional, Union
import pyflame as pf
import pyflame.nn as nn


class InferenceEngine:
    """Optimized inference engine for production deployment.

    Features:
    - Automatic batching
    - Input validation
    - Output caching
    - Performance metrics

    Example:
        >>> engine = InferenceEngine(model)
        >>> engine.warmup(example_input)
        >>> output = engine.infer(input_data)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_batch_size: int = 32,
        enable_caching: bool = False,
        cache_size: int = 1000,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.enable_caching = enable_caching
        self.cache_size = cache_size

        self._cache: Dict[int, pf.Tensor] = {}
        self._inference_count = 0
        self._total_time = 0.0

        # Set model to eval mode
        self.model.eval()

    def warmup(
        self,
        example_input: Union[pf.Tensor, List[pf.Tensor]],
        num_iterations: int = 10,
    ):
        """Warmup the model for optimal performance.

        Args:
            example_input: Example input for warmup
            num_iterations: Number of warmup iterations
        """
        with pf.no_grad():
            for _ in range(num_iterations):
                if isinstance(example_input, list):
                    self.model(*example_input)
                else:
                    self.model(example_input)

    def infer(
        self,
        inputs: Union[pf.Tensor, List[pf.Tensor]],
    ) -> Union[pf.Tensor, List[pf.Tensor]]:
        """Run inference on inputs.

        Args:
            inputs: Input tensor(s)

        Returns:
            Model output(s)
        """
        import time

        start_time = time.perf_counter()

        with pf.no_grad():
            if isinstance(inputs, list):
                outputs = self.model(*inputs)
            else:
                outputs = self.model(inputs)

        end_time = time.perf_counter()
        self._inference_count += 1
        self._total_time += end_time - start_time

        return outputs

    def batch_infer(
        self,
        inputs: List[pf.Tensor],
    ) -> List[pf.Tensor]:
        """Run batched inference on multiple inputs.

        Automatically batches inputs for efficiency.
        """
        results = []

        for i in range(0, len(inputs), self.max_batch_size):
            batch = inputs[i:i + self.max_batch_size]
            batched_input = pf.stack(batch, dim=0)
            batch_output = self.infer(batched_input)

            # Unbatch outputs
            for j in range(batch_output.shape[0]):
                results.append(batch_output[j])

        return results

    @property
    def stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = self._total_time / max(self._inference_count, 1)
        return {
            "inference_count": self._inference_count,
            "total_time_seconds": self._total_time,
            "average_time_ms": avg_time * 1000,
            "throughput_per_second": 1.0 / max(avg_time, 1e-10),
        }


def optimize_for_inference(
    model: nn.Module,
    example_input: pf.Tensor,
    *,
    fuse_operations: bool = True,
    constant_folding: bool = True,
    precision: str = "fp32",
) -> nn.Module:
    """Optimize model for inference.

    Applies various optimizations:
    - Operation fusion
    - Constant folding
    - Precision conversion
    - Dead code elimination

    Args:
        model: Model to optimize
        example_input: Example input for tracing
        fuse_operations: Enable operator fusion
        constant_folding: Enable constant folding
        precision: Target precision ("fp32", "fp16", "bf16")

    Returns:
        Optimized model
    """
    model.eval()

    # Trace model
    with pf.no_grad():
        traced = pf.trace(model, example_input)

    # Apply optimizations
    graph = traced.graph

    if fuse_operations:
        graph = pf.optimize.fuse_operations(graph)

    if constant_folding:
        graph = pf.optimize.fold_constants(graph)

    if precision != "fp32":
        dtype = {"fp16": pf.float16, "bf16": pf.bfloat16}[precision]
        graph = pf.optimize.convert_precision(graph, dtype)

    # Create optimized module
    optimized = pf.nn.OptimizedModule(graph)

    return optimized
```

### 7.3 Docker Containerization

```dockerfile
# docker/Dockerfile

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Build wheel
RUN pip wheel --no-deps --wheel-dir /wheels .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel from builder
COPY --from=builder /wheels/*.whl /wheels/

# Install PyFlame
RUN pip install /wheels/*.whl && rm -rf /wheels

# Copy serving code
COPY docker/serve.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "serve.py"]
```

```python
# docker/serve.py

"""Simple model serving script for Docker deployment."""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pyflame as pf
from pyflame.serving import InferenceEngine

app = FastAPI(title="PyFlame Model Server")

# Load model
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.pf")
model = None
engine = None


class PredictRequest(BaseModel):
    inputs: List[List[float]]


class PredictResponse(BaseModel):
    outputs: List[List[float]]
    inference_time_ms: float


@app.on_event("startup")
async def load_model():
    global model, engine

    # Load model
    state_dict = pf.load(MODEL_PATH)
    # Assuming model architecture is known
    model = create_model()
    model.load_state_dict(state_dict)

    # Create inference engine
    engine = InferenceEngine(model)

    # Warmup
    example = pf.randn([1, *get_input_shape()])
    engine.warmup(example)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import time

    # Convert inputs to tensor
    inputs = pf.tensor(request.inputs)

    # Run inference
    start = time.perf_counter()
    outputs = engine.infer(inputs)
    end = time.perf_counter()

    return PredictResponse(
        outputs=outputs.numpy().tolist(),
        inference_time_ms=(end - start) * 1000,
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/stats")
async def stats():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine.stats


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 8. Benchmarking & Performance Analysis

### 8.1 Benchmark Suite

```python
# python/pyflame/benchmarks/__init__.py

"""PyFlame benchmark suite."""

from .runner import BenchmarkRunner, benchmark
from .models import get_benchmark_model
from .results import BenchmarkResults, compare_results


__all__ = [
    "BenchmarkRunner",
    "benchmark",
    "get_benchmark_model",
    "BenchmarkResults",
    "compare_results",
]
```

```python
# python/pyflame/benchmarks/runner.py

"""Benchmark runner for PyFlame."""

import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
import pyflame as pf


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 64])
    precision: str = "fp32"
    profile_memory: bool = True


@dataclass
class BenchmarkResult:
    """Results from a single benchmark."""
    name: str
    batch_size: int
    latency_ms: float
    throughput: float
    memory_mb: float
    std_dev_ms: float


class BenchmarkRunner:
    """Run benchmarks on PyFlame models.

    Example:
        >>> runner = BenchmarkRunner()
        >>> results = runner.run_model_benchmark(
        ...     "resnet50",
        ...     model=resnet50(),
        ...     input_shape=[3, 224, 224]
        ... )
        >>> runner.print_results(results)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []

    def run_model_benchmark(
        self,
        name: str,
        model: pf.nn.Module,
        input_shape: List[int],
    ) -> List[BenchmarkResult]:
        """Benchmark a model across different batch sizes."""
        model.eval()
        results = []

        for batch_size in self.config.batch_sizes:
            result = self._benchmark_batch_size(
                name, model, input_shape, batch_size
            )
            results.append(result)
            self.results.append(result)

        return results

    def _benchmark_batch_size(
        self,
        name: str,
        model: pf.nn.Module,
        input_shape: List[int],
        batch_size: int,
    ) -> BenchmarkResult:
        """Benchmark model at specific batch size."""
        full_shape = [batch_size] + input_shape
        x = pf.randn(full_shape)

        # Warmup
        with pf.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = model(x)

        # Benchmark
        times = []
        with pf.no_grad():
            for _ in range(self.config.benchmark_iterations):
                start = time.perf_counter()
                _ = model(x)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        import statistics

        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        throughput = batch_size / (avg_time / 1000)  # samples/sec

        # Memory profiling
        memory_mb = 0
        if self.config.profile_memory:
            memory_mb = pf.get_memory_allocated() / (1024 * 1024)

        return BenchmarkResult(
            name=name,
            batch_size=batch_size,
            latency_ms=avg_time,
            throughput=throughput,
            memory_mb=memory_mb,
            std_dev_ms=std_dev,
        )

    def run_operation_benchmark(
        self,
        name: str,
        operation: Callable,
        input_shapes: List[List[int]],
    ) -> BenchmarkResult:
        """Benchmark a single operation."""
        inputs = [pf.randn(shape) for shape in input_shapes]

        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = operation(*inputs)

        # Benchmark
        times = []
        for _ in range(self.config.benchmark_iterations):
            start = time.perf_counter()
            _ = operation(*inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        import statistics

        return BenchmarkResult(
            name=name,
            batch_size=input_shapes[0][0] if input_shapes else 1,
            latency_ms=statistics.mean(times),
            throughput=1000 / statistics.mean(times),
            memory_mb=pf.get_memory_allocated() / (1024 * 1024),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        )

    def print_results(self, results: Optional[List[BenchmarkResult]] = None):
        """Print benchmark results in table format."""
        results = results or self.results

        print("\n" + "=" * 80)
        print("PyFlame Benchmark Results")
        print("=" * 80)
        print(f"{'Model':<20} {'Batch':<8} {'Latency (ms)':<15} "
              f"{'Throughput':<15} {'Memory (MB)':<12}")
        print("-" * 80)

        for r in results:
            print(f"{r.name:<20} {r.batch_size:<8} "
                  f"{r.latency_ms:>10.2f} +/- {r.std_dev_ms:.2f}  "
                  f"{r.throughput:>10.1f}/s  {r.memory_mb:>10.1f}")

        print("=" * 80 + "\n")

    def export_json(self, path: str):
        """Export results to JSON."""
        import json

        data = [
            {
                "name": r.name,
                "batch_size": r.batch_size,
                "latency_ms": r.latency_ms,
                "throughput": r.throughput,
                "memory_mb": r.memory_mb,
                "std_dev_ms": r.std_dev_ms,
            }
            for r in self.results
        ]

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def benchmark(
    model: pf.nn.Module,
    input_shape: List[int],
    batch_sizes: List[int] = [1, 8, 32],
    iterations: int = 100,
) -> List[BenchmarkResult]:
    """Convenience function for quick benchmarking.

    Example:
        >>> results = pf.benchmark(model, [3, 224, 224])
    """
    config = BenchmarkConfig(
        benchmark_iterations=iterations,
        batch_sizes=batch_sizes,
    )
    runner = BenchmarkRunner(config)
    return runner.run_model_benchmark("model", model, input_shape)
```

---

## 9. Plugin & Extension System

### 9.1 Overview

Phase 4 introduces a plugin system allowing users to extend PyFlame with custom operators, backends, and integrations.

### 9.2 Custom Operator Registration

```python
# python/pyflame/extend/__init__.py

"""PyFlame extension API."""

from .custom_op import register_custom_op, custom_op
from .plugin import Plugin, load_plugin, register_plugin


__all__ = [
    "register_custom_op",
    "custom_op",
    "Plugin",
    "load_plugin",
    "register_plugin",
]
```

```python
# python/pyflame/extend/custom_op.py

"""Custom operator registration for PyFlame."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import pyflame as pf


# Registry of custom operators
_custom_ops: Dict[str, "CustomOp"] = {}


class CustomOp:
    """Custom operator definition."""

    def __init__(
        self,
        name: str,
        forward_fn: Callable,
        backward_fn: Optional[Callable] = None,
        csl_template: Optional[str] = None,
    ):
        self.name = name
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn
        self.csl_template = csl_template

    def __call__(self, *args, **kwargs):
        return self.forward_fn(*args, **kwargs)


def register_custom_op(
    name: str,
    forward_fn: Callable,
    backward_fn: Optional[Callable] = None,
    csl_template: Optional[str] = None,
) -> CustomOp:
    """Register a custom operator.

    Args:
        name: Unique name for the operator
        forward_fn: Forward computation function
        backward_fn: Backward gradient function (for autograd)
        csl_template: Optional CSL code template for Cerebras execution

    Returns:
        Registered custom operator

    Example:
        >>> def my_activation(x):
        ...     return x * pf.sigmoid(x)  # Swish/SiLU
        ...
        >>> def my_activation_backward(grad_output, x):
        ...     sig = pf.sigmoid(x)
        ...     return grad_output * (sig + x * sig * (1 - sig))
        ...
        >>> swish = register_custom_op(
        ...     "swish",
        ...     forward_fn=my_activation,
        ...     backward_fn=my_activation_backward,
        ... )
        >>> y = swish(x)
    """
    op = CustomOp(name, forward_fn, backward_fn, csl_template)
    _custom_ops[name] = op

    # Register with C++ backend
    pf._register_custom_op(name, forward_fn, backward_fn, csl_template)

    return op


def custom_op(
    name: str,
    backward_fn: Optional[Callable] = None,
    csl_template: Optional[str] = None,
):
    """Decorator for registering custom operators.

    Example:
        >>> @custom_op("my_relu6")
        ... def relu6(x):
        ...     return pf.clamp(pf.relu(x), max=6.0)
        ...
        >>> y = relu6(x)
    """
    def decorator(fn: Callable) -> CustomOp:
        return register_custom_op(name, fn, backward_fn, csl_template)
    return decorator


def get_custom_op(name: str) -> Optional[CustomOp]:
    """Get registered custom operator by name."""
    return _custom_ops.get(name)


def list_custom_ops() -> List[str]:
    """List all registered custom operators."""
    return list(_custom_ops.keys())
```

### 9.3 Plugin System

```python
# python/pyflame/extend/plugin.py

"""Plugin system for PyFlame extensions."""

from typing import Any, Dict, List, Optional, Type
from abc import ABC, abstractmethod
import importlib
import pyflame as pf


class Plugin(ABC):
    """Base class for PyFlame plugins.

    Plugins can extend PyFlame with:
    - Custom operators
    - New backends
    - Data loaders
    - Integrations

    Example:
        >>> class MyPlugin(Plugin):
        ...     name = "my-plugin"
        ...     version = "1.0.0"
        ...
        ...     def setup(self):
        ...         # Register custom operators
        ...         register_custom_op("my_op", self.my_op_impl)
        ...
        ...     def teardown(self):
        ...         pass
        ...
        ...     def my_op_impl(self, x):
        ...         return x * 2
    """

    name: str = "unnamed-plugin"
    version: str = "0.0.0"
    description: str = ""
    dependencies: List[str] = []

    @abstractmethod
    def setup(self):
        """Initialize the plugin. Called when plugin is loaded."""
        pass

    @abstractmethod
    def teardown(self):
        """Cleanup the plugin. Called when plugin is unloaded."""
        pass

    def get_custom_ops(self) -> Dict[str, Any]:
        """Return custom operators provided by this plugin."""
        return {}

    def get_integrations(self) -> Dict[str, Any]:
        """Return integrations provided by this plugin."""
        return {}


# Plugin registry
_plugins: Dict[str, Plugin] = {}


def register_plugin(plugin_class: Type[Plugin]) -> Type[Plugin]:
    """Decorator to register a plugin class.

    Example:
        >>> @register_plugin
        ... class MyPlugin(Plugin):
        ...     name = "my-plugin"
        ...     ...
    """
    plugin = plugin_class()
    _plugins[plugin.name] = plugin
    return plugin_class


def load_plugin(name: str) -> Plugin:
    """Load and initialize a plugin by name.

    Args:
        name: Plugin name or module path

    Returns:
        Loaded plugin instance

    Example:
        >>> plugin = load_plugin("pyflame-wandb")
        >>> plugin = load_plugin("my_package.my_plugin:MyPlugin")
    """
    if name in _plugins:
        plugin = _plugins[name]
    else:
        # Try to import as module
        if ":" in name:
            module_path, class_name = name.rsplit(":", 1)
        else:
            module_path = name
            class_name = "Plugin"

        module = importlib.import_module(module_path)
        plugin_class = getattr(module, class_name)
        plugin = plugin_class()
        _plugins[plugin.name] = plugin

    # Check dependencies
    for dep in plugin.dependencies:
        if dep not in _plugins:
            raise RuntimeError(f"Plugin '{name}' requires '{dep}'")

    # Initialize
    plugin.setup()

    return plugin


def unload_plugin(name: str):
    """Unload a plugin.

    Args:
        name: Plugin name
    """
    if name in _plugins:
        plugin = _plugins[name]
        plugin.teardown()
        del _plugins[name]


def list_plugins() -> List[Dict[str, Any]]:
    """List all loaded plugins."""
    return [
        {
            "name": p.name,
            "version": p.version,
            "description": p.description,
        }
        for p in _plugins.values()
    ]
```

---

## 10. Implementation Roadmap

### 10.1 Milestones

| Milestone | Deliverable | Duration |
|-----------|-------------|----------|
| **M4.1** | VSCode extension (basic) | Weeks 1-3 |
| **M4.2** | Debugger and profiler | Weeks 4-6 |
| **M4.3** | Graph visualization tools | Weeks 7-8 |
| **M4.4** | Documentation system (Sphinx) | Weeks 9-10 |
| **M4.5** | PyPI distribution setup | Weeks 11-12 |
| **M4.6** | ONNX import/export | Weeks 13-15 |
| **M4.7** | MLOps integrations (W&B, MLflow) | Weeks 16-17 |
| **M4.8** | Jupyter integration | Weeks 18-19 |
| **M4.9** | Model serving infrastructure | Weeks 20-22 |
| **M4.10** | Benchmark suite | Weeks 23-24 |
| **M4.11** | Plugin system | Weeks 25-26 |

### 10.2 Dependencies

```
       Developer Tools (M4.1-M4.3)
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
Docs      Distribution   Integrations
(M4.4)    (M4.5)        (M4.6-M4.8)
   │          │              │
   └──────────┼──────────────┘
              ▼
       Production Deployment
           (M4.9)
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
Benchmarks  Plugin     Community
(M4.10)     System     Infrastructure
            (M4.11)
```

### 10.3 Testing Strategy

1. **Developer Tools**: Manual testing with real IDE workflows
2. **Documentation**: Link checking, example execution verification
3. **Distribution**: Install testing on clean environments
4. **Integrations**: Integration tests with third-party services
5. **Deployment**: End-to-end serving tests
6. **Benchmarks**: Regression testing for performance

---

## 11. Technical Decisions

### 11.1 Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| IDE Extension | VSCode first | Largest ML developer user base |
| Documentation | Sphinx + Furo | Industry standard, good Jupyter support |
| Build System | scikit-build-core | Modern, cross-platform wheel building |
| ONNX Version | Opset 17 | Good coverage of modern ops |
| MLOps Integration | W&B + MLflow | Most widely used platforms |
| Serving Framework | FastAPI | Modern, async, good performance |

### 11.2 Open Questions

1. **PyCharm Support**: Timeline for PyCharm plugin?
2. **Cloud Deployment**: AWS/GCP/Azure specific integrations?
3. **Model Zoo Hosting**: Self-hosted vs. cloud storage for pretrained weights?
4. **Plugin Marketplace**: Community plugin distribution mechanism?

### 11.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| IDE integration complexity | Medium | Medium | Prioritize core features, iterate |
| ONNX compatibility gaps | High | Medium | Focus on common ops, document limitations |
| Distribution issues | Low | High | Extensive testing on multiple platforms |
| Community adoption | Medium | High | Good documentation, examples, support |

---

## Appendix A: Tools Module Structure

```
python/pyflame/tools/
├── __init__.py
├── debugger.py          # Interactive debugger
├── profiler.py          # Performance profiler
├── visualization.py     # Graph visualization
├── language_server.py   # LSP implementation
└── utils.py             # Utility functions
```

---

## Appendix B: Example Plugin Structure

```
pyflame-plugin-example/
├── pyproject.toml
├── README.md
├── src/
│   └── pyflame_plugin_example/
│       ├── __init__.py
│       ├── plugin.py       # Plugin class definition
│       ├── ops.py          # Custom operators
│       └── integrations.py # Third-party integrations
└── tests/
    └── test_plugin.py
```

---

*Document Version: 1.0*
*Last Updated: January 11, 2026*
