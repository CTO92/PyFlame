"""
PyFlame Graph Visualization tools.

Provides computation graph visualization and export capabilities.
"""

import html
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Maximum DOT string length to prevent DoS
MAX_DOT_LENGTH = 10 * 1024 * 1024  # 10 MB

# Timeout for Graphviz subprocess (seconds)
GRAPHVIZ_TIMEOUT = 60


@dataclass
class NodeStyle:
    """Style configuration for graph nodes."""

    fill_color: str = "#4a90d9"
    border_color: str = "#2c5aa0"
    text_color: str = "#ffffff"
    font_size: int = 12
    shape: str = "box"  # box, ellipse, diamond


@dataclass
class EdgeStyle:
    """Style configuration for graph edges."""

    color: str = "#666666"
    width: float = 1.0
    style: str = "solid"  # solid, dashed, dotted


class GraphVisualizer:
    """Visualizer for PyFlame computation graphs.

    Generates DOT format graphs that can be rendered with Graphviz
    or converted to SVG/PNG.

    Example:
        >>> viz = GraphVisualizer(graph)
        >>> viz.to_dot("graph.dot")
        >>> viz.to_svg("graph.svg")
    """

    # Node colors by operation category
    OP_COLORS = {
        "input": "#6ab04c",  # Green
        "output": "#e17055",  # Red
        "matmul": "#4a90d9",  # Blue
        "conv": "#4a90d9",  # Blue
        "activation": "#f9ca24",  # Yellow
        "norm": "#9b59b6",  # Purple
        "pool": "#1abc9c",  # Teal
        "loss": "#e74c3c",  # Red
        "elementwise": "#95a5a6",  # Gray
        "default": "#34495e",  # Dark gray
    }

    def __init__(
        self,
        graph: Optional[Any] = None,
        show_shapes: bool = True,
        show_dtypes: bool = True,
        show_values: bool = False,
        max_nodes: int = 500,
        rankdir: str = "TB",
    ):
        """Initialize visualizer.

        Args:
            graph: PyFlame Graph object
            show_shapes: Display tensor shapes
            show_dtypes: Display data types
            show_values: Display tensor values (small tensors only)
            max_nodes: Maximum nodes to render
            rankdir: Graph direction (TB=top-bottom, LR=left-right)
        """
        self.graph = graph
        self.show_shapes = show_shapes
        self.show_dtypes = show_dtypes
        self.show_values = show_values
        self.max_nodes = max_nodes
        self.rankdir = rankdir

        self._nodes: List[Dict[str, Any]] = []
        self._edges: List[Tuple[int, int, str]] = []
        self._node_styles: Dict[int, NodeStyle] = {}
        self._edge_styles: Dict[Tuple[int, int], EdgeStyle] = {}

    def _get_op_category(self, op_name: str) -> str:
        """Get the category for an operation.

        Args:
            op_name: Operation name

        Returns:
            Category string
        """
        op_lower = op_name.lower()

        if "input" in op_lower or "const" in op_lower:
            return "input"
        if "output" in op_lower:
            return "output"
        if "matmul" in op_lower or "mm" in op_lower or "linear" in op_lower:
            return "matmul"
        if "conv" in op_lower:
            return "conv"
        if any(
            act in op_lower for act in ["relu", "sigmoid", "tanh", "gelu", "softmax"]
        ):
            return "activation"
        if any(
            norm in op_lower for norm in ["norm", "bn", "ln", "layernorm", "batchnorm"]
        ):
            return "norm"
        if "pool" in op_lower:
            return "pool"
        if "loss" in op_lower or "criterion" in op_lower:
            return "loss"
        if any(ew in op_lower for ew in ["add", "sub", "mul", "div", "neg"]):
            return "elementwise"

        return "default"

    def _get_node_color(self, op_name: str) -> str:
        """Get color for a node based on operation type.

        Args:
            op_name: Operation name

        Returns:
            Hex color string
        """
        category = self._get_op_category(op_name)
        return self.OP_COLORS.get(category, self.OP_COLORS["default"])

    def _format_shape(self, shape: List[int]) -> str:
        """Format a tensor shape for display.

        Args:
            shape: Tensor shape

        Returns:
            Formatted string
        """
        return "[" + ", ".join(str(s) for s in shape) + "]"

    def _build_graph_data(self):
        """Build internal graph representation from PyFlame graph."""
        if self.graph is None:
            return

        self._nodes = []
        self._edges = []

        # Get nodes from graph
        try:
            # Try to iterate over graph nodes
            node_count = 0
            visited: Set[int] = set()

            # This is a placeholder - actual implementation depends on Graph API
            if hasattr(self.graph, "nodes"):
                for node in self.graph.nodes:
                    if node_count >= self.max_nodes:
                        break

                    node_id = id(node) if not hasattr(node, "id") else node.id
                    if node_id in visited:
                        continue
                    visited.add(node_id)

                    op_name = (
                        node.op if hasattr(node, "op") else str(type(node).__name__)
                    )
                    shape = node.shape if hasattr(node, "shape") else None
                    dtype = node.dtype if hasattr(node, "dtype") else None

                    self._nodes.append(
                        {
                            "id": node_id,
                            "op": op_name,
                            "shape": shape,
                            "dtype": dtype,
                        }
                    )

                    # Get edges from inputs
                    if hasattr(node, "inputs"):
                        for input_node in node.inputs:
                            input_id = (
                                id(input_node)
                                if not hasattr(input_node, "id")
                                else input_node.id
                            )
                            self._edges.append((input_id, node_id, ""))

                    node_count += 1

            elif hasattr(self.graph, "get_nodes"):
                nodes = self.graph.get_nodes()
                for node in nodes[: self.max_nodes]:
                    node_id = node.id if hasattr(node, "id") else id(node)
                    self._nodes.append(
                        {
                            "id": node_id,
                            "op": getattr(node, "op", "unknown"),
                            "shape": getattr(node, "shape", None),
                            "dtype": getattr(node, "dtype", None),
                        }
                    )

        except Exception:
            # Fallback for unknown graph format
            pass

    def to_dot(self, path: Optional[str] = None) -> str:
        """Generate DOT format graph.

        Args:
            path: Optional file path to write to

        Returns:
            DOT format string
        """
        self._build_graph_data()

        lines = [
            "digraph PyFlameGraph {",
            f"    rankdir={self.rankdir};",
            '    node [shape=box, style=filled, fontname="Arial", fontsize=10];',
            '    edge [fontname="Arial", fontsize=9];',
            "",
        ]

        # Add nodes
        for node in self._nodes:
            node_id = node["id"]
            op = node["op"]
            color = self._get_node_color(op)

            # Build label
            label_parts = [html.escape(op)]
            if self.show_shapes and node.get("shape"):
                label_parts.append(f'shape: {self._format_shape(node["shape"])}')
            if self.show_dtypes and node.get("dtype"):
                label_parts.append(f'dtype: {node["dtype"]}')

            label = "\\n".join(label_parts)
            lines.append(
                f'    node_{node_id} [label="{label}", fillcolor="{color}", '
                f'fontcolor="white"];'
            )

        lines.append("")

        # Add edges
        for src, dst, label in self._edges:
            edge_label = f' [label="{label}"]' if label else ""
            lines.append(f"    node_{src} -> node_{dst}{edge_label};")

        lines.append("}")

        dot_string = "\n".join(lines)

        if path:
            with open(path, "w") as f:
                f.write(dot_string)

        return dot_string

    def _sanitize_dot_string(self, dot_string: str) -> str:
        """Sanitize DOT string to remove potentially dangerous content.

        Args:
            dot_string: Raw DOT format string

        Returns:
            Sanitized DOT string
        """
        # Check length limit
        if len(dot_string) > MAX_DOT_LENGTH:
            raise ValueError(
                f"DOT string too large ({len(dot_string)} bytes). "
                f"Maximum allowed: {MAX_DOT_LENGTH} bytes"
            )

        # Remove potentially dangerous attributes that could cause issues
        # Graphviz supports URL/href attributes that could be exploited
        dangerous_patterns = [
            r"URL\s*=",
            r"href\s*=",
            r"target\s*=",
            r"onclick\s*=",
            r"onmouseover\s*=",
            r"<script",
            r"javascript:",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, dot_string, re.IGNORECASE):
                logger.warning(
                    f"Removed potentially dangerous pattern from DOT: {pattern}"
                )
                dot_string = re.sub(
                    pattern + r"[^;,\]]*", "", dot_string, flags=re.IGNORECASE
                )

        return dot_string

    def to_svg(self, path: Optional[str] = None) -> str:
        """Generate SVG visualization.

        Requires graphviz to be installed.

        Args:
            path: Optional file path to write to

        Returns:
            SVG string
        """
        dot_string = self.to_dot()

        # Sanitize the DOT string before passing to Graphviz
        dot_string = self._sanitize_dot_string(dot_string)

        try:
            import subprocess

            result = subprocess.run(
                ["dot", "-Tsvg"],
                input=dot_string,
                capture_output=True,
                text=True,
                check=True,
                timeout=GRAPHVIZ_TIMEOUT,  # Prevent hanging
            )
            svg_string = result.stdout

            if path:
                with open(path, "w") as f:
                    f.write(svg_string)

            return svg_string

        except FileNotFoundError:
            raise RuntimeError(
                "Graphviz 'dot' command not found. "
                "Please install Graphviz: https://graphviz.org/download/"
            )
        except subprocess.TimeoutExpired:
            logger.error("Graphviz process timed out")
            raise RuntimeError(
                f"Graphviz process timed out after {GRAPHVIZ_TIMEOUT} seconds. "
                "The graph may be too complex."
            )
        except subprocess.CalledProcessError as e:
            # Don't expose full stderr to prevent information leakage
            logger.error(f"Graphviz error: {e.stderr}")
            raise RuntimeError("Graphviz failed to process the graph")

    def to_svg_string(self) -> str:
        """Generate SVG as a string without saving to file.

        Returns:
            SVG string
        """
        return self.to_svg()

    def to_png(self, path: str, dpi: int = 150):
        """Generate PNG visualization.

        Requires graphviz to be installed.

        Args:
            path: File path to write to
            dpi: Image DPI
        """
        dot_string = self.to_dot()

        # Sanitize the DOT string before passing to Graphviz
        dot_string = self._sanitize_dot_string(dot_string)

        try:
            import subprocess

            subprocess.run(
                ["dot", "-Tpng", f"-Gdpi={dpi}", "-o", path],
                input=dot_string,
                text=True,
                check=True,
                timeout=GRAPHVIZ_TIMEOUT,  # Prevent hanging
            )

        except FileNotFoundError:
            raise RuntimeError(
                "Graphviz 'dot' command not found. "
                "Please install Graphviz: https://graphviz.org/download/"
            )
        except subprocess.TimeoutExpired:
            logger.error("Graphviz process timed out during PNG generation")
            raise RuntimeError(
                f"Graphviz process timed out after {GRAPHVIZ_TIMEOUT} seconds. "
                "The graph may be too complex."
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Graphviz error during PNG generation: {e}")
            raise RuntimeError("Graphviz failed to process the graph")

    def to_html(self, path: Optional[str] = None, title: str = "PyFlame Graph") -> str:
        """Generate interactive HTML visualization.

        Args:
            path: Optional file path to write to
            title: Page title

        Returns:
            HTML string
        """
        svg = self.to_svg()

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 100%;
            overflow: auto;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
        }}
        .graph-container {{
            text-align: center;
            overflow: auto;
        }}
        svg {{
            max-width: 100%;
            height: auto;
        }}
        .legend {{
            margin-top: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 5px 10px;
            background: #f0f0f0;
            border-radius: 4px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{html.escape(title)}</h1>
        <div class="graph-container">
            {svg}
        </div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: {self.OP_COLORS['input']};"></div>
                <span>Input</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {self.OP_COLORS['matmul']};"></div>
                <span>Matrix Ops</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {self.OP_COLORS['activation']};"></div>
                <span>Activation</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {self.OP_COLORS['norm']};"></div>
                <span>Normalization</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {self.OP_COLORS['output']};"></div>
                <span>Output</span>
            </div>
        </div>
    </div>
</body>
</html>"""

        if path:
            with open(path, "w") as f:
                f.write(html_content)

        return html_content


def visualize_graph(
    graph,
    output_path: Optional[str] = None,
    format: str = "svg",
    **kwargs,
) -> Union[str, None]:
    """Convenience function to visualize a computation graph.

    Args:
        graph: PyFlame Graph object
        output_path: Optional output file path
        format: Output format ("dot", "svg", "png", "html")
        **kwargs: Additional arguments for GraphVisualizer

    Returns:
        Visualization string (for dot/svg/html) or None (for png)

    Example:
        >>> visualize_graph(model_graph, "model.svg")
        >>> dot_string = visualize_graph(model_graph, format="dot")
    """
    viz = GraphVisualizer(graph, **kwargs)

    if format == "dot":
        return viz.to_dot(output_path)
    elif format == "svg":
        return viz.to_svg(output_path)
    elif format == "png":
        if output_path is None:
            raise ValueError("output_path required for PNG format")
        viz.to_png(output_path)
        return None
    elif format == "html":
        return viz.to_html(output_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def visualize_model(
    model,
    example_input,
    output_path: Optional[str] = None,
    format: str = "svg",
    **kwargs,
) -> Union[str, None]:
    """Visualize a model by tracing it with example input.

    Args:
        model: PyFlame model (nn.Module)
        example_input: Example input tensor
        output_path: Optional output file path
        format: Output format
        **kwargs: Additional arguments

    Returns:
        Visualization string or None
    """
    # Run model to build graph
    output = model(example_input)

    # Get graph from output
    try:
        import pyflame as pf

        graph = pf.get_graph(output)
    except Exception:
        graph = None

    if graph is None:
        raise ValueError("Could not extract computation graph from model output")

    return visualize_graph(graph, output_path, format, **kwargs)
