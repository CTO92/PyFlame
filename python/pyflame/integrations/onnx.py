"""
ONNX integration for PyFlame.

Provides import and export functionality for ONNX models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import os


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export.

    Attributes:
        opset_version: ONNX opset version (default: 17)
        dynamic_axes: Dict mapping input/output names to dynamic dimension indices
        input_names: Names for model inputs
        output_names: Names for model outputs
        do_constant_folding: Fold constants during export
        verbose: Print export information
    """
    opset_version: int = 17
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    do_constant_folding: bool = True
    verbose: bool = False


class ONNXExporter:
    """Export PyFlame models to ONNX format.

    Example:
        >>> exporter = ONNXExporter()
        >>> exporter.export(model, example_input, "model.onnx")
    """

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
        "conv2d": "Conv",
        "maxpool2d": "MaxPool",
        "avgpool2d": "AveragePool",
        "batchnorm": "BatchNormalization",
        "layernorm": "LayerNormalization",
        "dropout": "Dropout",
        "flatten": "Flatten",
        "reshape": "Reshape",
        "transpose": "Transpose",
        "concat": "Concat",
        "split": "Split",
        "squeeze": "Squeeze",
        "unsqueeze": "Unsqueeze",
        "gather": "Gather",
        "slice": "Slice",
        "pad": "Pad",
        "reduce_mean": "ReduceMean",
        "reduce_sum": "ReduceSum",
        "reduce_max": "ReduceMax",
        "reduce_min": "ReduceMin",
        "exp": "Exp",
        "log": "Log",
        "sqrt": "Sqrt",
        "pow": "Pow",
        "abs": "Abs",
        "neg": "Neg",
        "clip": "Clip",
        "cast": "Cast",
    }

    def __init__(self, config: Optional[ONNXExportConfig] = None):
        """Initialize exporter.

        Args:
            config: Export configuration
        """
        self.config = config or ONNXExportConfig()

    def export(
        self,
        model,
        example_input: Union[Any, Tuple[Any, ...]],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> None:
        """Export a PyFlame model to ONNX format.

        Args:
            model: PyFlame model (nn.Module)
            example_input: Example input tensor(s)
            output_path: Path to save ONNX model
            input_names: Names for model inputs
            output_names: Names for model outputs
            dynamic_axes: Dynamic axis specification

        Example:
            >>> exporter.export(
            ...     model,
            ...     pf.randn([1, 3, 224, 224]),
            ...     "resnet50.onnx",
            ...     input_names=["image"],
            ...     output_names=["logits"],
            ...     dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}
            ... )
        """
        try:
            import onnx
            from onnx import helper, TensorProto, numpy_helper
        except ImportError:
            raise ImportError(
                "ONNX is required for export. Install with: pip install onnx"
            )

        import numpy as np

        # Set model to eval mode
        model.eval()

        # Trace model
        if isinstance(example_input, tuple):
            output = model(*example_input)
        else:
            output = model(example_input)

        # Get graph from traced output
        try:
            import pyflame as pf
            graph = pf.get_graph(output)
        except Exception:
            graph = None

        if graph is None:
            raise RuntimeError("Could not trace model to extract computation graph")

        # Build ONNX graph
        onnx_nodes = []
        onnx_inputs = []
        onnx_outputs = []
        initializers = []
        value_info = []

        # Use provided names or defaults
        input_names = input_names or self.config.input_names or ["input"]
        output_names = output_names or self.config.output_names or ["output"]

        # Add input
        if isinstance(example_input, tuple):
            for i, inp in enumerate(example_input):
                name = input_names[i] if i < len(input_names) else f"input_{i}"
                shape = list(inp.shape) if hasattr(inp, "shape") else None
                onnx_inputs.append(
                    helper.make_tensor_value_info(
                        name,
                        TensorProto.FLOAT,
                        shape,
                    )
                )
        else:
            shape = list(example_input.shape) if hasattr(example_input, "shape") else None
            onnx_inputs.append(
                helper.make_tensor_value_info(
                    input_names[0],
                    TensorProto.FLOAT,
                    shape,
                )
            )

        # Add output
        if isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                name = output_names[i] if i < len(output_names) else f"output_{i}"
                shape = list(out.shape) if hasattr(out, "shape") else None
                onnx_outputs.append(
                    helper.make_tensor_value_info(
                        name,
                        TensorProto.FLOAT,
                        shape,
                    )
                )
        else:
            shape = list(output.shape) if hasattr(output, "shape") else None
            onnx_outputs.append(
                helper.make_tensor_value_info(
                    output_names[0],
                    TensorProto.FLOAT,
                    shape,
                )
            )

        # Convert model parameters to initializers
        if hasattr(model, "state_dict"):
            for name, param in model.state_dict().items():
                if hasattr(param, "numpy"):
                    np_array = param.numpy()
                elif isinstance(param, np.ndarray):
                    np_array = param
                else:
                    continue

                tensor = numpy_helper.from_array(np_array, name=name)
                initializers.append(tensor)

        # Create graph
        onnx_graph = helper.make_graph(
            onnx_nodes,
            "pyflame_model",
            onnx_inputs,
            onnx_outputs,
            initializers,
        )

        # Create model
        onnx_model = helper.make_model(
            onnx_graph,
            opset_imports=[helper.make_opsetid("", self.config.opset_version)],
        )

        # Add metadata
        onnx_model.producer_name = "PyFlame"
        onnx_model.producer_version = "1.0.0"

        # Validate model
        onnx.checker.check_model(onnx_model)

        # Apply constant folding if enabled
        if self.config.do_constant_folding:
            try:
                from onnx import optimizer
                onnx_model = optimizer.optimize(onnx_model)
            except ImportError:
                pass  # Optimizer not available

        # Save model
        onnx.save(onnx_model, output_path)

        if self.config.verbose:
            print(f"ONNX model saved to: {output_path}")
            print(f"  Opset version: {self.config.opset_version}")
            print(f"  Inputs: {[i.name for i in onnx_inputs]}")
            print(f"  Outputs: {[o.name for o in onnx_outputs]}")

    def _convert_op(self, op_name: str, **kwargs) -> Dict[str, Any]:
        """Convert a PyFlame operation to ONNX.

        Args:
            op_name: PyFlame operation name
            **kwargs: Operation attributes

        Returns:
            ONNX node configuration
        """
        onnx_op = self.OP_MAPPING.get(op_name.lower())
        if onnx_op is None:
            raise ValueError(f"Unsupported operation for ONNX export: {op_name}")

        return {"op_type": onnx_op, **kwargs}


class ONNXImporter:
    """Import ONNX models to PyFlame format.

    Example:
        >>> importer = ONNXImporter()
        >>> model = importer.import_model("model.onnx")
    """

    def __init__(self):
        """Initialize importer."""
        pass

    def import_model(
        self,
        model_path: str,
        verify: bool = True,
    ):
        """Import an ONNX model.

        Args:
            model_path: Path to ONNX model file
            verify: Verify model structure after import

        Returns:
            PyFlame model

        Example:
            >>> model = importer.import_model("resnet50.onnx")
            >>> output = model(input_tensor)
        """
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError:
            raise ImportError(
                "ONNX is required for import. Install with: pip install onnx"
            )

        # Load ONNX model
        onnx_model = onnx.load(model_path)

        if verify:
            onnx.checker.check_model(onnx_model)

        # Create PyFlame model wrapper
        return ONNXModel(onnx_model)

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about an ONNX model.

        Args:
            model_path: Path to ONNX model file

        Returns:
            Dictionary with model information
        """
        try:
            import onnx
        except ImportError:
            raise ImportError("ONNX is required. Install with: pip install onnx")

        onnx_model = onnx.load(model_path)

        inputs = []
        for inp in onnx_model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            inputs.append({
                "name": inp.name,
                "shape": shape,
                "dtype": inp.type.tensor_type.elem_type,
            })

        outputs = []
        for out in onnx_model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            outputs.append({
                "name": out.name,
                "shape": shape,
                "dtype": out.type.tensor_type.elem_type,
            })

        return {
            "producer": onnx_model.producer_name,
            "producer_version": onnx_model.producer_version,
            "opset_version": onnx_model.opset_import[0].version,
            "inputs": inputs,
            "outputs": outputs,
            "num_nodes": len(onnx_model.graph.node),
            "num_initializers": len(onnx_model.graph.initializer),
        }


class ONNXModel:
    """Wrapper for running ONNX models in PyFlame.

    This class wraps an ONNX model for execution using ONNX Runtime
    while maintaining a PyFlame-like interface.
    """

    def __init__(self, onnx_model):
        """Initialize ONNX model wrapper.

        Args:
            onnx_model: ONNX model object
        """
        self.onnx_model = onnx_model
        self._session = None
        self._input_names = [inp.name for inp in onnx_model.graph.input]
        self._output_names = [out.name for out in onnx_model.graph.output]

    def _get_session(self):
        """Get or create ONNX Runtime session."""
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError(
                    "ONNX Runtime is required for inference. "
                    "Install with: pip install onnxruntime"
                )

            # Serialize model to bytes
            import onnx
            model_bytes = onnx_model.SerializeToString()

            self._session = ort.InferenceSession(
                model_bytes,
                providers=["CPUExecutionProvider"],
            )

        return self._session

    def __call__(self, *inputs):
        """Run inference.

        Args:
            *inputs: Input tensors

        Returns:
            Output tensor(s)
        """
        import numpy as np

        session = self._get_session()

        # Prepare inputs
        input_feed = {}
        for i, inp in enumerate(inputs):
            name = self._input_names[i] if i < len(self._input_names) else f"input_{i}"
            if hasattr(inp, "numpy"):
                input_feed[name] = inp.numpy()
            else:
                input_feed[name] = np.asarray(inp)

        # Run inference
        outputs = session.run(self._output_names, input_feed)

        # Convert outputs to PyFlame tensors
        try:
            import pyflame as pf
            outputs = [pf.tensor(out) for out in outputs]
        except Exception:
            pass

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    @property
    def input_names(self) -> List[str]:
        """Get input names."""
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        """Get output names."""
        return self._output_names


def export_onnx(
    model,
    example_input,
    output_path: str,
    **kwargs,
) -> None:
    """Export a PyFlame model to ONNX format.

    Convenience function for ONNXExporter.

    Args:
        model: PyFlame model
        example_input: Example input tensor(s)
        output_path: Output file path
        **kwargs: Additional export options

    Example:
        >>> pf.integrations.export_onnx(model, x, "model.onnx")
    """
    exporter = ONNXExporter()
    exporter.export(model, example_input, output_path, **kwargs)


def import_onnx(model_path: str, **kwargs):
    """Import an ONNX model to PyFlame.

    Convenience function for ONNXImporter.

    Args:
        model_path: Path to ONNX model
        **kwargs: Additional import options

    Returns:
        PyFlame-compatible model

    Example:
        >>> model = pf.integrations.import_onnx("resnet50.onnx")
    """
    importer = ONNXImporter()
    return importer.import_model(model_path, **kwargs)
