"""
Custom operator registration for PyFlame.

Allows users to define and register custom operations.
"""

import hashlib
import inspect
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Security: Maximum allowed template size (1 MB)
_MAX_TEMPLATE_SIZE = 1024 * 1024

# Security: Pattern for dangerous CSL constructs that could indicate injection
_DANGEROUS_CSL_PATTERNS = [
    r"@import_module\s*\(\s*[\"'][^<]",  # Non-system module imports
    r"@comptime_print",  # Debug output that could leak info
    r"@panic",  # Could be used for DoS
    r"__asm__",  # Inline assembly
    r"@ptrCast",  # Pointer manipulation
    r"@intToPtr",  # Integer to pointer
    r"@ptrToInt",  # Pointer to integer
    r"@bitCast",  # Bit-level type punning
    r"@alignCast",  # Alignment manipulation
    r"extern\s+fn",  # External function declarations
    r"@cImport",  # C imports
    r"@embedFile",  # File embedding
    r"std\.os\.",  # OS-level operations
    r"std\.fs\.",  # Filesystem operations
    r"std\.net\.",  # Network operations
    r"std\.process\.",  # Process operations
    r"std\.child_process\.",  # Child process operations
    r"std\.debug\.",  # Debug operations
    r"std\.heap\.",  # Heap manipulation
    r"@import\s*\(\s*[\"']std[\"']\s*\)",  # Full std import
    r"unreachable",  # Could indicate malicious control flow
    r"@setRuntimeSafety\s*\(\s*false",  # Disabling safety checks
]

# Suspicious strings that may indicate malicious intent
_SUSPICIOUS_STRINGS = [
    "system(",
    "exec(",
    "eval(",
    "shell",
    "/bin/",
    "/etc/",
    "/proc/",
    "/dev/",
    "/tmp/",
    "password",
    "secret",
    "token",
    "credential",
    "api_key",
    "private_key",
    "ssh_key",
    "env[",
    "environ",
    "getenv",
    "curl ",
    "wget ",
    "chmod ",
    "base64",
    "\\x",  # Hex escape sequences
    "fromhex",
]

# Zero-width and invisible characters that could hide malicious code
_INVISIBLE_CHARS = frozenset(
    [
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u2060",  # Word joiner
        "\u2061",  # Function application
        "\u2062",  # Invisible times
        "\u2063",  # Invisible separator
        "\u2064",  # Invisible plus
        "\ufeff",  # Zero-width no-break space (BOM)
        "\u00ad",  # Soft hyphen
        "\u034f",  # Combining grapheme joiner
        "\u061c",  # Arabic letter mark
        "\u115f",  # Hangul choseong filler
        "\u1160",  # Hangul jungseong filler
        "\u17b4",  # Khmer vowel inherent aq
        "\u17b5",  # Khmer vowel inherent aa
        "\u180e",  # Mongolian vowel separator
    ]
)


def _normalize_unicode(text: str) -> str:
    """Normalize Unicode to detect homoglyph attacks.

    Converts confusable characters to their ASCII equivalents where possible.
    """
    # NFKC normalization converts compatibility characters to canonical form
    normalized = unicodedata.normalize("NFKC", text)

    # Additional mapping for common homoglyphs that survive NFKC
    homoglyph_map = {
        "\u0391": "A",  # Greek Alpha
        "\u0392": "B",  # Greek Beta
        "\u0395": "E",  # Greek Epsilon
        "\u0396": "Z",  # Greek Zeta
        "\u0397": "H",  # Greek Eta
        "\u0399": "I",  # Greek Iota
        "\u039a": "K",  # Greek Kappa
        "\u039c": "M",  # Greek Mu
        "\u039d": "N",  # Greek Nu
        "\u039f": "O",  # Greek Omicron
        "\u03a1": "P",  # Greek Rho
        "\u03a4": "T",  # Greek Tau
        "\u03a7": "X",  # Greek Chi
        "\u03a5": "Y",  # Greek Upsilon
        "\u0410": "A",  # Cyrillic A
        "\u0412": "B",  # Cyrillic Ve
        "\u0415": "E",  # Cyrillic Ie
        "\u041a": "K",  # Cyrillic Ka
        "\u041c": "M",  # Cyrillic Em
        "\u041d": "H",  # Cyrillic En
        "\u041e": "O",  # Cyrillic O
        "\u0420": "P",  # Cyrillic Er
        "\u0421": "C",  # Cyrillic Es
        "\u0422": "T",  # Cyrillic Te
        "\u0425": "X",  # Cyrillic Ha
        "\u0430": "a",  # Cyrillic a
        "\u0435": "e",  # Cyrillic ie
        "\u043e": "o",  # Cyrillic o
        "\u0440": "p",  # Cyrillic er
        "\u0441": "c",  # Cyrillic es
        "\u0443": "y",  # Cyrillic u
        "\u0445": "x",  # Cyrillic ha
        "\u2010": "-",  # Hyphen
        "\u2011": "-",  # Non-breaking hyphen
        "\u2012": "-",  # Figure dash
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\uff20": "@",  # Fullwidth @
        "\uff3f": "_",  # Fullwidth _
    }

    result = []
    for char in normalized:
        result.append(homoglyph_map.get(char, char))

    return "".join(result)


def _validate_csl_template(template: str, op_name: str) -> None:
    """Validate a CSL template for security issues.

    Args:
        template: The CSL template string
        op_name: Name of the operation (for error messages)

    Raises:
        ValueError: If the template contains potentially dangerous constructs

    Security Note:
        This validation provides defense-in-depth but is not exhaustive.
        CSL templates are code that will be compiled and run on Cerebras
        hardware. Only use templates from trusted sources.
    """
    if not template:
        return

    # Check template size
    if len(template) > _MAX_TEMPLATE_SIZE:
        raise ValueError(
            f"CSL template for '{op_name}' exceeds maximum size "
            f"({len(template)} > {_MAX_TEMPLATE_SIZE} bytes)"
        )

    # Security: Check for invisible/zero-width characters that could hide code
    invisible_found = [c for c in template if c in _INVISIBLE_CHARS]
    if invisible_found:
        logger.error(
            f"SECURITY: CSL template for '{op_name}' contains {len(invisible_found)} "
            f"invisible/zero-width characters which could hide malicious code"
        )
        raise ValueError(
            f"CSL template for '{op_name}' contains invisible characters. "
            f"These could be used to hide malicious code."
        )

    # Security: Check for control characters (except newline, tab, carriage return)
    control_chars = [c for c in template if ord(c) < 32 and c not in "\n\t\r"]
    if control_chars:
        logger.error(
            f"SECURITY: CSL template for '{op_name}' contains {len(control_chars)} "
            f"control characters"
        )
        raise ValueError(
            f"CSL template for '{op_name}' contains invalid control characters"
        )

    # Security: Normalize Unicode to detect homoglyph attacks
    # This converts look-alike characters (e.g., Cyrillic 'а' -> ASCII 'a')
    normalized_template = _normalize_unicode(template)

    # Check for null bytes which could cause string truncation
    if "\x00" in template or "\x00" in normalized_template:
        logger.error(f"SECURITY: CSL template for '{op_name}' contains null bytes")
        raise ValueError(f"CSL template for '{op_name}' contains invalid null bytes")

    # Security: Check for multi-line obfuscation (keywords split across lines)
    # Remove all whitespace to detect split keywords
    compressed = re.sub(r"\s+", "", normalized_template.lower())
    split_keyword_checks = [
        "importunsafe",
        "cimport",
        "embedfile",
        "ptrcast",
        "importmodule",
    ]
    for keyword in split_keyword_checks:
        if keyword in compressed:
            logger.warning(
                f"SECURITY: CSL template for '{op_name}' may contain obfuscated "
                f"keyword '{keyword}' (possibly split across lines)"
            )
            raise ValueError(
                f"CSL template for '{op_name}' contains suspicious patterns. "
                f"Only use CSL templates from trusted sources."
            )

    # Check for dangerous patterns in both original and normalized
    for check_template in [template, normalized_template]:
        for pattern in _DANGEROUS_CSL_PATTERNS:
            if re.search(pattern, check_template, re.IGNORECASE | re.MULTILINE):
                logger.warning(
                    f"SECURITY: CSL template for '{op_name}' contains potentially "
                    f"dangerous pattern matching '{pattern}'. Template rejected."
                )
                raise ValueError(
                    f"CSL template for '{op_name}' contains potentially unsafe constructs. "
                    f"Only use CSL templates from trusted sources."
                )

    # Check for suspicious string patterns
    template_lower = normalized_template.lower()
    for suspicious in _SUSPICIOUS_STRINGS:
        if suspicious in template_lower:
            logger.warning(
                f"SECURITY: CSL template for '{op_name}' contains suspicious "
                f"string '{suspicious}'."
            )
            raise ValueError(
                f"CSL template for '{op_name}' contains suspicious content. "
                f"Only use CSL templates from trusted sources."
            )

    # Check for non-ASCII characters that survived normalization
    # (may indicate obfuscation attempts)
    non_ascii_chars = [c for c in normalized_template if ord(c) > 127]
    if non_ascii_chars:
        unique_non_ascii = set(non_ascii_chars)
        if len(unique_non_ascii) > 10:  # Allow some Unicode in comments/strings
            logger.warning(
                f"SECURITY: CSL template for '{op_name}' contains many "
                f"non-ASCII characters which may indicate obfuscation"
            )
            raise ValueError(
                f"CSL template for '{op_name}' contains suspicious non-ASCII content"
            )


@dataclass
class CSLTemplateInfo:
    """Security metadata for CSL template tracking.

    Attributes:
        source_file: File path where the template was defined (if available)
        source_line: Line number where the template was defined
        registered_at: Unix timestamp when the template was registered
        sha256_hash: SHA256 hash of the template content for integrity verification
        validated: Whether the template passed security validation
        validation_warnings: Any warnings generated during validation
    """

    source_file: Optional[str] = None
    source_line: Optional[int] = None
    registered_at: float = field(default_factory=time.time)
    sha256_hash: Optional[str] = None
    validated: bool = False
    validation_warnings: List[str] = field(default_factory=list)


def _compute_template_hash(template: str) -> str:
    """Compute SHA256 hash of a CSL template for integrity tracking."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _get_caller_info() -> Tuple[Optional[str], Optional[int]]:
    """Get the file and line number of the caller for source tracking.

    Security: This helps track where CSL templates originate from for auditing.
    """
    try:
        # Walk up the stack to find the first frame outside this module
        for frame_info in inspect.stack():
            if frame_info.filename != __file__:
                return frame_info.filename, frame_info.lineno
    except Exception:
        pass
    return None, None


@dataclass
class CustomOp:
    """Custom operator definition.

    Attributes:
        name: Unique operator name
        forward_fn: Forward computation function
        backward_fn: Backward gradient function (optional)
        csl_template: CSL code template for Cerebras execution (optional).
            SECURITY WARNING: CSL templates are code that will be compiled
            and executed. Only use templates from trusted sources.
        schema: Operator schema definition
        doc: Documentation string
        csl_template_info: Security metadata for CSL template tracking

    Security Note:
        The csl_template field accepts arbitrary CSL code. This code will be
        compiled by the Cerebras SDK and executed on hardware. Only use
        templates from sources you trust completely.

        The csl_template_info field tracks the origin and validation status
        of CSL templates for security auditing purposes.
    """

    name: str
    forward_fn: Callable
    backward_fn: Optional[Callable] = None
    csl_template: Optional[str] = None
    schema: Optional[str] = None
    doc: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    csl_template_info: Optional[CSLTemplateInfo] = None

    def __call__(self, *args, **kwargs):
        """Execute the custom operator."""
        return self.forward_fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"CustomOp(name={self.name!r})"


# Global registry of custom operators
_custom_ops: Dict[str, CustomOp] = {}


def register_custom_op(
    name: str,
    forward_fn: Callable,
    backward_fn: Optional[Callable] = None,
    csl_template: Optional[str] = None,
    schema: Optional[str] = None,
    doc: str = "",
    **metadata,
) -> CustomOp:
    """Register a custom operator.

    Creates and registers a new operator that can be used in PyFlame
    computation graphs.

    Args:
        name: Unique operator name
        forward_fn: Forward computation function
        backward_fn: Backward gradient function for autograd
        csl_template: CSL code template for Cerebras execution.
            SECURITY WARNING: Templates are compiled and executed.
            Only use templates from trusted sources.
        schema: Operator schema (e.g., "(Tensor x, Tensor y) -> Tensor")
        doc: Documentation string
        **metadata: Additional metadata

    Returns:
        Registered CustomOp instance

    Raises:
        ValueError: If the operator name is already registered or if the
            CSL template fails security validation.

    Example:
        >>> def swish_forward(x):
        ...     return x * pf.sigmoid(x)
        ...
        >>> def swish_backward(grad_output, x):
        ...     sig = pf.sigmoid(x)
        ...     return grad_output * (sig + x * sig * (1 - sig))
        ...
        >>> swish = register_custom_op(
        ...     "swish",
        ...     forward_fn=swish_forward,
        ...     backward_fn=swish_backward,
        ...     doc="Swish/SiLU activation function"
        ... )
        >>> y = swish(x)
    """
    if name in _custom_ops:
        raise ValueError(f"Custom op '{name}' already registered")

    # Security: Track CSL template source and validation
    csl_info = None
    if csl_template:
        # Get caller information for security auditing
        source_file, source_line = _get_caller_info()

        # Validate the template
        validation_warnings = []
        _validate_csl_template(csl_template, name)

        # Create security metadata
        csl_info = CSLTemplateInfo(
            source_file=source_file,
            source_line=source_line,
            registered_at=time.time(),
            sha256_hash=_compute_template_hash(csl_template),
            validated=True,
            validation_warnings=validation_warnings,
        )

        logger.info(
            f"Registering custom op '{name}' with CSL template. "
            f"Source: {source_file}:{source_line}, "
            f"Hash: {csl_info.sha256_hash[:16]}... "
            f"Ensure the template is from a trusted source."
        )

    op = CustomOp(
        name=name,
        forward_fn=forward_fn,
        backward_fn=backward_fn,
        csl_template=csl_template,
        schema=schema,
        doc=doc,
        metadata=metadata,
        csl_template_info=csl_info,
    )

    _custom_ops[name] = op

    # Try to register with C++ backend if available
    try:
        import pyflame as pf

        if hasattr(pf, "_register_custom_op"):
            pf._register_custom_op(name, forward_fn, backward_fn, csl_template)
    except (ImportError, AttributeError):
        pass

    return op


def custom_op(
    name: str,
    backward_fn: Optional[Callable] = None,
    csl_template: Optional[str] = None,
    schema: Optional[str] = None,
    **metadata,
):
    """Decorator for registering custom operators.

    Convenient way to register a function as a custom operator.

    Args:
        name: Unique operator name
        backward_fn: Backward gradient function
        csl_template: CSL code template
        schema: Operator schema
        **metadata: Additional metadata

    Returns:
        Decorator function

    Example:
        >>> @custom_op("my_relu6")
        ... def relu6(x):
        ...     return pf.clamp(pf.relu(x), max=6.0)
        ...
        >>> y = relu6(x)

        >>> @custom_op("my_gelu", schema="(Tensor x) -> Tensor")
        ... def my_gelu(x):
        ...     return x * pf.sigmoid(1.702 * x)
    """

    def decorator(fn: Callable) -> CustomOp:
        return register_custom_op(
            name=name,
            forward_fn=fn,
            backward_fn=backward_fn,
            csl_template=csl_template,
            schema=schema,
            doc=fn.__doc__ or "",
            **metadata,
        )

    return decorator


def get_custom_op(name: str) -> Optional[CustomOp]:
    """Get a registered custom operator by name.

    Args:
        name: Operator name

    Returns:
        CustomOp instance or None if not found
    """
    return _custom_ops.get(name)


def list_custom_ops() -> List[str]:
    """List all registered custom operators.

    Returns:
        List of operator names
    """
    return list(_custom_ops.keys())


def unregister_custom_op(name: str) -> bool:
    """Unregister a custom operator.

    Args:
        name: Operator name

    Returns:
        True if removed, False if not found
    """
    if name in _custom_ops:
        del _custom_ops[name]
        return True
    return False


def clear_custom_ops():
    """Clear all registered custom operators."""
    _custom_ops.clear()


class AutogradFunction:
    """Base class for autograd-compatible custom operations.

    Provides a more structured way to define custom operations with
    automatic differentiation support.

    Example:
        >>> class MySoftplus(AutogradFunction):
        ...     @staticmethod
        ...     def forward(ctx, x, beta=1.0):
        ...         ctx.save_for_backward(x)
        ...         ctx.beta = beta
        ...         return (1 / beta) * pf.log(1 + pf.exp(beta * x))
        ...
        ...     @staticmethod
        ...     def backward(ctx, grad_output):
        ...         x, = ctx.saved_tensors
        ...         beta = ctx.beta
        ...         return grad_output * pf.sigmoid(beta * x)
        ...
        >>> softplus = MySoftplus.apply
        >>> y = softplus(x, beta=2.0)
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Forward pass.

        Args:
            ctx: Context object for saving tensors
            *args: Input arguments
            **kwargs: Keyword arguments

        Returns:
            Operation output
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass.

        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient of the output

        Returns:
            Gradients with respect to inputs
        """
        raise NotImplementedError("Subclasses must implement backward()")

    @classmethod
    def apply(cls, *args, **kwargs):
        """Apply the function.

        This is the main entry point for using the function.
        """
        ctx = _FunctionContext()
        output = cls.forward(ctx, *args, **kwargs)

        # Store context for backward pass
        if hasattr(output, "_autograd_ctx"):
            output._autograd_ctx = (cls, ctx)

        return output


class _FunctionContext:
    """Context object for storing tensors during forward pass."""

    def __init__(self):
        self.saved_tensors: Tuple = ()
        self._saved_data: Dict[str, Any] = {}

    def save_for_backward(self, *tensors):
        """Save tensors for the backward pass.

        Args:
            *tensors: Tensors to save
        """
        self.saved_tensors = tensors

    def __setattr__(self, name: str, value: Any):
        if name in ("saved_tensors", "_saved_data"):
            super().__setattr__(name, value)
        else:
            self._saved_data[name] = value

    def __getattr__(self, name: str) -> Any:
        if name == "_saved_data":
            return super().__getattribute__("_saved_data")
        return self._saved_data.get(name)


# Common custom operations
def _register_builtin_ops():
    """Register common custom operations."""

    # Swish/SiLU activation
    @custom_op("swish", schema="(Tensor x) -> Tensor")
    def swish(x):
        """Swish/SiLU activation: x * sigmoid(x)"""
        try:
            import pyflame as pf

            return x * pf.sigmoid(x)
        except ImportError:
            import numpy as np

            return x * (1 / (1 + np.exp(-x)))

    # Mish activation
    @custom_op("mish", schema="(Tensor x) -> Tensor")
    def mish(x):
        """Mish activation: x * tanh(softplus(x))"""
        try:
            import pyflame as pf

            return x * pf.tanh(pf.log(1 + pf.exp(x)))
        except ImportError:
            import numpy as np

            return x * np.tanh(np.log(1 + np.exp(x)))

    # Hard Swish
    @custom_op("hard_swish", schema="(Tensor x) -> Tensor")
    def hard_swish(x):
        """Hard Swish activation: x * relu6(x + 3) / 6"""
        try:
            import pyflame as pf

            return x * pf.clamp(x + 3, min=0, max=6) / 6
        except ImportError:
            import numpy as np

            return x * np.clip(x + 3, 0, 6) / 6

    # GELU approximation
    @custom_op("gelu_approx", schema="(Tensor x) -> Tensor")
    def gelu_approx(x):
        """Fast GELU approximation: x * sigmoid(1.702 * x)"""
        try:
            import pyflame as pf

            return x * pf.sigmoid(1.702 * x)
        except ImportError:
            import numpy as np

            return x * (1 / (1 + np.exp(-1.702 * x)))


# Register built-in ops on module load
try:
    _register_builtin_ops()
except Exception:
    pass  # Ignore errors during import
