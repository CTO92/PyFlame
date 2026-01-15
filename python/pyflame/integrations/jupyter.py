"""
Jupyter notebook integration for PyFlame.

Provides rich tensor display, graph visualization, and progress bars.
"""

import html
from typing import Any, Dict, Optional


def setup_jupyter():
    """Setup PyFlame for Jupyter notebooks.

    Enables:
    - Rich tensor display with statistics
    - Graph visualization
    - Interactive progress bars
    - Automatic output formatting

    Call this at the start of a notebook to enable PyFlame integration.

    Example:
        >>> import pyflame as pf
        >>> pf.integrations.setup_jupyter()
        PyFlame Jupyter integration enabled
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            print("Not running in IPython/Jupyter environment")
            return

        # Get HTML formatter
        formatter = ip.display_formatter.formatters.get("text/html")
        if formatter is None:
            print("HTML formatter not available")
            return

        # Try to import pyflame and register formatters
        try:
            import pyflame as pf

            # Register tensor formatter
            formatter.for_type(pf.Tensor, _tensor_to_html)

            # Register graph formatter if available
            if hasattr(pf, "Graph"):
                formatter.for_type(pf.Graph, _graph_to_html)

        except ImportError:
            pass

        print("PyFlame Jupyter integration enabled")

    except ImportError:
        print("IPython not available. Jupyter integration requires IPython.")


def _tensor_to_html(tensor) -> str:
    """Convert PyFlame tensor to HTML for Jupyter display.

    Args:
        tensor: PyFlame Tensor object

    Returns:
        HTML string representation
    """
    # Get tensor properties
    shape = list(tensor.shape) if hasattr(tensor, "shape") else []
    shape_str = " x ".join(str(s) for s in shape)
    dtype_str = str(tensor.dtype) if hasattr(tensor, "dtype") else "unknown"
    numel = tensor.numel if hasattr(tensor, "numel") else 0
    is_evaluated = tensor.is_evaluated() if hasattr(tensor, "is_evaluated") else True

    # Status indicator
    status_color = "#2ecc71" if is_evaluated else "#f39c12"
    status_text = "Evaluated" if is_evaluated else "Lazy"

    html_parts = [
        '<div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; '
        "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
        "background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%); "
        'box-shadow: 0 2px 4px rgba(0,0,0,0.05);">',
        # Header
        '<div style="display: flex; align-items: center; margin-bottom: 10px;">',
        '<span style="font-weight: 600; color: #2c3e50; font-size: 14px;">PyFlame Tensor</span>',
        f'<span style="margin-left: auto; background: {status_color}; color: white; '
        f'padding: 2px 8px; border-radius: 4px; font-size: 11px;">{status_text}</span>',
        "</div>",
        # Properties table
        '<table style="width: 100%; border-collapse: collapse; font-size: 13px;">',
        "<tr>",
        '<td style="padding: 4px 8px; color: #666;">Shape</td>',
        f'<td style="padding: 4px 8px; font-weight: 500; color: #2c3e50;">[{shape_str}]</td>',
        "</tr>",
        '<tr style="background: #f8f9fa;">',
        '<td style="padding: 4px 8px; color: #666;">DType</td>',
        f'<td style="padding: 4px 8px; font-weight: 500; color: #2c3e50;">{dtype_str}</td>',
        "</tr>",
        "<tr>",
        '<td style="padding: 4px 8px; color: #666;">Elements</td>',
        f'<td style="padding: 4px 8px; font-weight: 500; color: #2c3e50;">{numel:,}</td>',
        "</tr>",
    ]

    # Add statistics if tensor is small and evaluated
    if is_evaluated and numel <= 10000:
        try:
            import numpy as np

            data = tensor.numpy()

            stats = {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
            }

            html_parts.extend(
                [
                    '<tr style="background: #f8f9fa;">',
                    '<td style="padding: 4px 8px; color: #666;">Stats</td>',
                    '<td style="padding: 4px 8px; font-family: monospace; font-size: 12px;">',
                    f'min={stats["min"]:.4f}, max={stats["max"]:.4f}, ',
                    f'mean={stats["mean"]:.4f}, std={stats["std"]:.4f}',
                    "</td>",
                    "</tr>",
                ]
            )

            # Check for NaN/Inf
            has_nan = np.isnan(data).any()
            has_inf = np.isinf(data).any()
            if has_nan or has_inf:
                warnings = []
                if has_nan:
                    warnings.append("NaN")
                if has_inf:
                    warnings.append("Inf")
                html_parts.extend(
                    [
                        "<tr>",
                        '<td style="padding: 4px 8px; color: #e74c3c;">Warning</td>',
                        f'<td style="padding: 4px 8px; color: #e74c3c;">Contains {", ".join(warnings)}</td>',
                        "</tr>",
                    ]
                )

        except Exception:
            pass

    html_parts.append("</table>")

    # Show data preview for small tensors
    if is_evaluated and numel <= 100:
        try:
            import numpy as np

            data = tensor.numpy()
            data_str = np.array2string(
                data, precision=4, suppress_small=True, max_line_width=80
            )
            html_parts.extend(
                [
                    '<div style="margin-top: 10px; padding: 8px; background: #2c3e50; '
                    'border-radius: 4px; overflow-x: auto;">',
                    f'<pre style="margin: 0; color: #ecf0f1; font-size: 12px; '
                    f"font-family: 'Monaco', 'Menlo', monospace;\">{html.escape(data_str)}</pre>",
                    "</div>",
                ]
            )
        except Exception:
            pass

    html_parts.append("</div>")

    return "".join(html_parts)


def _graph_to_html(graph) -> str:
    """Convert PyFlame computation graph to HTML visualization.

    Args:
        graph: PyFlame Graph object

    Returns:
        HTML string with embedded SVG visualization
    """
    try:
        from pyflame.tools.visualization import GraphVisualizer

        viz = GraphVisualizer(graph, max_nodes=100)
        svg = viz.to_svg_string()

        return f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #fff;">
            <div style="font-weight: 600; color: #2c3e50; margin-bottom: 10px;">
                PyFlame Computation Graph
            </div>
            <div style="max-height: 600px; overflow: auto; border: 1px solid #eee;
                        border-radius: 4px; padding: 10px; background: #fafafa;">
                {svg}
            </div>
        </div>
        """
    except Exception as e:
        return f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px;
                    color: #e74c3c;">
            Could not render graph: {html.escape(str(e))}
        </div>
        """


class TensorWidget:
    """Interactive tensor widget for Jupyter.

    Provides an interactive view of tensor data with:
    - Dimension slicing
    - Statistics
    - Histogram visualization

    Example:
        >>> widget = TensorWidget(tensor)
        >>> display(widget)
    """

    def __init__(self, tensor, name: str = "Tensor"):
        """Initialize widget.

        Args:
            tensor: PyFlame tensor
            name: Display name
        """
        self.tensor = tensor
        self.name = name
        self._widget = None

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        return _tensor_to_html(self.tensor)

    def show_histogram(self, bins: int = 50):
        """Display histogram of tensor values.

        Args:
            bins: Number of histogram bins
        """
        try:
            import matplotlib.pyplot as plt

            data = self.tensor.numpy().flatten()

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(data, bins=bins, edgecolor="white", alpha=0.7)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.set_title(f"{self.name} Value Distribution")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib required for histogram visualization")
        except Exception as e:
            print(f"Could not create histogram: {e}")

    def show_heatmap(self, cmap: str = "viridis"):
        """Display 2D tensor as heatmap.

        Args:
            cmap: Colormap name
        """
        try:
            import matplotlib.pyplot as plt

            data = self.tensor.numpy()

            if data.ndim != 2:
                print(f"Heatmap requires 2D tensor, got {data.ndim}D")
                return

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(data, cmap=cmap, aspect="auto")
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{self.name} Heatmap")
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib required for heatmap visualization")
        except Exception as e:
            print(f"Could not create heatmap: {e}")


class ProgressBar:
    """Progress bar for Jupyter notebooks.

    Provides a rich progress bar for training loops.

    Example:
        >>> progress = ProgressBar(total=100)
        >>> for i in range(100):
        ...     progress.update(i + 1, loss=0.5)
        >>> progress.close()
    """

    def __init__(
        self,
        total: int,
        desc: str = "",
        unit: str = "it",
    ):
        """Initialize progress bar.

        Args:
            total: Total iterations
            desc: Description text
            unit: Unit name
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self._display_id = None

        try:
            from IPython.display import HTML, display

            self._display = display
            self._HTML = HTML
            self._in_jupyter = True
        except ImportError:
            self._in_jupyter = False

        self._start_display()

    def _start_display(self):
        """Initialize display."""
        if self._in_jupyter:
            from IPython.display import display

            handle = display(self._render(), display_id=True)
            self._display_id = handle.display_id

    def _render(self, metrics: Optional[Dict[str, Any]] = None) -> Any:
        """Render progress bar HTML.

        Args:
            metrics: Optional metrics to display

        Returns:
            HTML object
        """
        if not self._in_jupyter:
            return None

        percent = (self.current / self.total * 100) if self.total > 0 else 0
        bar_width = min(percent, 100)

        metrics_str = ""
        if metrics:
            metrics_parts = [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ]
            metrics_str = " | ".join(metrics_parts)

        html = f"""
        <div style="font-family: monospace; font-size: 13px; margin: 5px 0;">
            <span style="color: #666;">{self.desc}</span>
            <div style="display: inline-block; width: 300px; height: 20px;
                        background: #eee; border-radius: 4px; overflow: hidden;
                        vertical-align: middle; margin: 0 10px;">
                <div style="width: {bar_width}%; height: 100%;
                            background: linear-gradient(90deg, #3498db, #2ecc71);
                            transition: width 0.2s;"></div>
            </div>
            <span>{self.current}/{self.total} [{percent:.0f}%]</span>
            <span style="color: #888; margin-left: 10px;">{metrics_str}</span>
        </div>
        """
        return self._HTML(html)

    def update(self, n: int = 1, **metrics):
        """Update progress bar.

        Args:
            n: New current value (or increment if None)
            **metrics: Metrics to display
        """
        self.current = n

        if self._in_jupyter and self._display_id:
            from IPython.display import update_display

            update_display(self._render(metrics), display_id=self._display_id)
        else:
            # Fallback to console
            percent = (self.current / self.total * 100) if self.total > 0 else 0
            print(f"\r{self.desc} {self.current}/{self.total} [{percent:.0f}%]", end="")

    def close(self):
        """Close and finalize progress bar."""
        if not self._in_jupyter:
            print()  # New line
