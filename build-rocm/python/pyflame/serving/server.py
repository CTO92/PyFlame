"""
Model serving server for PyFlame.

Provides HTTP/REST API for model inference.
"""

import asyncio
import hmac
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Security: Maximum warmup timeout in seconds
MAX_WARMUP_TIMEOUT_SECONDS = 120


@dataclass
class ServerConfig:
    """Configuration for model server.

    Attributes:
        host: Server host. Default is 127.0.0.1 (localhost only) for security.
            Set to "0.0.0.0" to accept connections from all interfaces.
        port: Server port
        workers: Number of worker processes
        timeout: Request timeout in seconds
        max_batch_size: Maximum batch size
        max_input_size_mb: Maximum input size in megabytes
        enable_cors: Enable CORS
        cors_origins: List of allowed CORS origins. SECURITY: Must be explicitly
            configured for production. Empty list disables CORS entirely.
            Default (None) allows localhost origins only for development.
        strict_cors: If True, reject wildcard ('*') CORS origins. SECURITY:
            Enable this for production to prevent overly permissive CORS.
        api_prefix: API route prefix
        api_key: Optional API key for authentication. SECURITY: Strongly
            recommended for production deployments.
        rate_limit_per_minute: Rate limit per client IP per minute (0 = disabled).
            SECURITY WARNING: This is per-worker, in-memory rate limiting only.
            It does NOT provide protection in multi-worker or distributed deployments.
            For production, use an external rate limiter (Redis, nginx, etc.).
            Disabled by default to avoid false sense of security.
        ssl_certfile: Path to SSL certificate file
        ssl_keyfile: Path to SSL key file
        warmup_input_shape: Shape of dummy input for model warmup (default: (1, 10))
        warmup_timeout_seconds: Maximum time for warmup operations (default: 60)
    """

    host: str = "127.0.0.1"  # Security: Default to localhost only
    port: int = 8000
    workers: int = 1
    timeout: int = 60
    max_batch_size: int = 32
    max_input_size_mb: float = 100.0
    enable_cors: bool = False  # Security: Disabled by default
    cors_origins: List[str] = None  # Must be explicitly configured
    strict_cors: bool = False  # Security: Enable for production
    api_prefix: str = "/v1"
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 0  # Disabled - use external rate limiter
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    warmup_input_shape: Tuple[int, ...] = (1, 10)
    warmup_timeout_seconds: int = 60  # Security: Timeout for warmup operations


class ModelServer:
    """HTTP server for PyFlame model inference.

    Provides a REST API for running model inference.

    Example:
        >>> server = ModelServer(model)
        >>> server.serve()  # Starts server on port 8000

        # Or with custom config
        >>> config = ServerConfig(port=8080, workers=4)
        >>> server = ModelServer(model, config=config)
        >>> server.serve()
    """

    def __init__(
        self,
        model,
        config: Optional[ServerConfig] = None,
        preprocess: Optional[Callable] = None,
        postprocess: Optional[Callable] = None,
    ):
        """Initialize server.

        Args:
            model: PyFlame model
            config: Server configuration
            preprocess: Optional input preprocessing function
            postprocess: Optional output postprocessing function
        """
        self.model = model
        self.config = config or ServerConfig()
        self.preprocess = preprocess
        self.postprocess = postprocess

        self._app = None
        self._engine = None

        # Set model to eval mode
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _create_app(self):
        """Create FastAPI application with security features."""
        try:
            from fastapi import Depends, FastAPI, HTTPException, Request
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.security import APIKeyHeader
            from pydantic import BaseModel
            from starlette.middleware.base import BaseHTTPMiddleware
        except ImportError:
            raise ImportError(
                "FastAPI is required for serving. "
                "Install with: pip install fastapi uvicorn"
            )

        from .inference import InferenceConfig, InferenceEngine

        app = FastAPI(
            title="PyFlame Model Server",
            description="REST API for PyFlame model inference",
            version="1.0.0",
        )

        # Rate limiting state with thread safety
        rate_limit_state = defaultdict(list)
        rate_limit_lock = threading.Lock()

        # Security middleware for headers
        class SecurityHeadersMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                response = await call_next(request)
                # Security headers
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["X-XSS-Protection"] = "1; mode=block"
                response.headers["Cache-Control"] = "no-store"
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains"
                )
                response.headers["Permissions-Policy"] = (
                    "geolocation=(), camera=(), microphone=()"
                )
                # Security: Content Security Policy - defense in depth
                response.headers["Content-Security-Policy"] = (
                    "default-src 'none'; "
                    "frame-ancestors 'none'; "
                    "base-uri 'none'; "
                    "form-action 'none'"
                )
                # Security: Referrer policy
                response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                return response

        app.add_middleware(SecurityHeadersMiddleware)

        # Request validation middleware for Content-Type and size limits
        max_input_mb = self.config.max_input_size_mb

        class RequestValidationMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Security: Validate Content-Type for POST requests
                if request.method == "POST":
                    content_type = request.headers.get("content-type", "")
                    # Allow JSON content type (with optional charset)
                    if not content_type.startswith("application/json"):
                        from starlette.responses import JSONResponse

                        return JSONResponse(
                            status_code=415,
                            content={"detail": "Content-Type must be application/json"},
                        )

                    # Security: Check Content-Length before processing
                    content_length = request.headers.get("content-length")
                    if content_length:
                        try:
                            size_bytes = int(content_length)
                            max_bytes = int(max_input_mb * 1024 * 1024)
                            if size_bytes > max_bytes:
                                return JSONResponse(
                                    status_code=413,
                                    content={
                                        "detail": f"Request body too large ({size_bytes} bytes). "
                                        f"Maximum: {max_bytes} bytes"
                                    },
                                )
                        except ValueError:
                            return JSONResponse(
                                status_code=400,
                                content={"detail": "Invalid Content-Length header"},
                            )

                return await call_next(request)

        app.add_middleware(RequestValidationMiddleware)

        # Enable CORS if configured with restricted origins
        if self.config.enable_cors:
            if self.config.cors_origins is None:
                # Development mode: allow localhost only with warning
                logger.warning(
                    "CORS enabled without explicit origins. Using localhost defaults. "
                    "For production, explicitly configure cors_origins in ServerConfig."
                )
                allowed_origins = [
                    "http://localhost:3000",
                    "http://localhost:8080",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:8080",
                ]
            elif not self.config.cors_origins:
                # Empty list provided - disable CORS
                logger.info("CORS origins list is empty; CORS disabled.")
                allowed_origins = None
            else:
                # Explicit origins configured
                allowed_origins = self.config.cors_origins
                # Security: Handle wildcard origins
                if "*" in allowed_origins:
                    if self.config.strict_cors:
                        # Strict mode: reject wildcard origins entirely
                        raise ValueError(
                            "SECURITY ERROR: Wildcard CORS origin ('*') is not allowed "
                            "when strict_cors=True. Please specify explicit origins or "
                            "set strict_cors=False (not recommended for production)."
                        )
                    else:
                        logger.warning(
                            "SECURITY WARNING: CORS allows all origins ('*'). "
                            "This is insecure for production deployments. "
                            "Enable strict_cors=True to reject wildcard origins."
                        )

            if allowed_origins:
                app.add_middleware(
                    CORSMiddleware,
                    allow_origins=allowed_origins,
                    allow_credentials=False,  # Don't allow credentials with CORS
                    allow_methods=["GET", "POST"],
                    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
                )

        # API key authentication
        api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

        async def verify_api_key(api_key: str = Depends(api_key_header)):
            if self.config.api_key is not None:
                # Use timing-safe comparison to prevent timing attacks
                if api_key is None or not hmac.compare_digest(
                    api_key.encode("utf-8"), self.config.api_key.encode("utf-8")
                ):
                    logger.warning("Failed API key authentication attempt")
                    raise HTTPException(
                        status_code=401, detail="Invalid or missing API key"
                    )
            return api_key

        # Rate limiting dependency with headers
        rate_limit_config = self.config.rate_limit_per_minute

        async def check_rate_limit(request: Request):
            if rate_limit_config <= 0:
                return {"limit": 0, "remaining": 0, "reset": 0}

            client_ip = request.client.host
            current_time = time.time()
            window_start = current_time - 60

            with rate_limit_lock:
                # Clean old entries
                rate_limit_state[client_ip] = [
                    t for t in rate_limit_state[client_ip] if t > window_start
                ]

                current_count = len(rate_limit_state[client_ip])
                remaining = max(0, rate_limit_config - current_count)
                reset_time = int(window_start + 60)

                if current_count >= rate_limit_config:
                    logger.warning(f"Rate limit exceeded for {client_ip}")
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded. Try again later.",
                        headers={
                            "X-RateLimit-Limit": str(rate_limit_config),
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(reset_time),
                            "Retry-After": str(reset_time - int(current_time)),
                        },
                    )

                rate_limit_state[client_ip].append(current_time)
                return {
                    "limit": rate_limit_config,
                    "remaining": remaining - 1,  # After this request
                    "reset": reset_time,
                }

        # Create inference engine
        engine_config = InferenceConfig(
            max_batch_size=self.config.max_batch_size,
        )
        self._engine = InferenceEngine(self.model, config=engine_config)

        # Define request/response models
        class PredictRequest(BaseModel):
            inputs: List[List[float]]
            batch: bool = False

        class PredictResponse(BaseModel):
            outputs: List[List[float]]
            inference_time_ms: float

        class HealthResponse(BaseModel):
            status: str
            model_loaded: bool

        class StatsResponse(BaseModel):
            total_inferences: int
            average_time_ms: float
            throughput_per_second: float

        prefix = self.config.api_prefix

        @app.get(f"{prefix}/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None,
            )

        @app.post(f"{prefix}/predict", response_model=PredictResponse)
        async def predict(
            request: PredictRequest,
            req: Request,
            _: str = Depends(verify_api_key),
            rate_info: dict = Depends(check_rate_limit),
        ):
            """Run model prediction with input validation."""
            from starlette.responses import JSONResponse

            try:
                import numpy as np

                # Input validation
                if not request.inputs:
                    raise HTTPException(status_code=400, detail="Empty inputs")

                if len(request.inputs) > self.config.max_batch_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Batch size {len(request.inputs)} exceeds maximum {self.config.max_batch_size}",
                    )

                # Convert inputs
                inputs = np.array(request.inputs, dtype=np.float32)

                # Check input size
                input_size_mb = inputs.nbytes / (1024 * 1024)
                if input_size_mb > self.config.max_input_size_mb:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input size {input_size_mb:.1f}MB exceeds maximum {self.config.max_input_size_mb}MB",
                    )

                # Check for invalid values
                if np.any(np.isnan(inputs)) or np.any(np.isinf(inputs)):
                    raise HTTPException(
                        status_code=400,
                        detail="Input contains invalid values (NaN or Inf)",
                    )

                # Apply preprocessing if defined
                if self.preprocess:
                    inputs = self.preprocess(inputs)

                # Convert to PyFlame tensor
                try:
                    import pyflame as pf

                    inputs = pf.tensor(inputs)
                except Exception:
                    pass

                # Run inference
                outputs, inference_time = self._engine.infer(inputs, return_time=True)

                # Apply postprocessing if defined
                if self.postprocess:
                    outputs = self.postprocess(outputs)

                # Convert outputs to list
                if hasattr(outputs, "numpy"):
                    outputs = outputs.numpy()
                outputs = np.asarray(outputs).tolist()

                # Log successful request
                logger.debug(
                    f"Prediction from {req.client.host}: {input_size_mb:.2f}MB, {inference_time:.2f}ms"
                )

                # Build response with rate limit headers
                response_data = PredictResponse(
                    outputs=outputs if isinstance(outputs[0], list) else [outputs],
                    inference_time_ms=inference_time,
                )

                response = JSONResponse(content=response_data.model_dump())

                # Add rate limit headers if rate limiting is enabled
                if rate_info["limit"] > 0:
                    response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
                    response.headers["X-RateLimit-Remaining"] = str(
                        rate_info["remaining"]
                    )
                    response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

                return response

            except HTTPException:
                raise  # Re-raise HTTP exceptions as-is
            except Exception as e:
                # Log the full error internally but return generic message
                logger.error(
                    f"Prediction error from {req.client.host}: {e}", exc_info=True
                )
                raise HTTPException(status_code=500, detail="Inference failed")

        @app.get(f"{prefix}/stats", response_model=StatsResponse)
        async def stats():
            """Get inference statistics."""
            s = self._engine.stats
            return StatsResponse(
                total_inferences=s.total_inferences,
                average_time_ms=s.average_time_ms,
                throughput_per_second=s.throughput,
            )

        # Warmup timeout from config (capped by global maximum)
        warmup_timeout = min(
            self.config.warmup_timeout_seconds, MAX_WARMUP_TIMEOUT_SECONDS
        )

        @app.post(f"{prefix}/warmup")
        async def warmup(
            iterations: int = 10,
            _: str = Depends(verify_api_key),
        ):
            """Warmup the model (requires authentication if API key is set)."""
            # Limit iterations to prevent DoS
            if iterations < 1 or iterations > 100:
                raise HTTPException(
                    status_code=400, detail="Iterations must be between 1 and 100"
                )

            async def do_warmup():
                import numpy as np

                # Create dummy input using configurable warmup shape
                dummy_input = np.random.randn(*self.config.warmup_input_shape).astype(
                    np.float32
                )

                try:
                    import pyflame as pf

                    dummy_input = pf.tensor(dummy_input)
                except Exception:
                    pass

                # Run warmup in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self._engine.warmup, dummy_input, iterations
                )
                return {
                    "status": "warmed up",
                    "iterations": iterations,
                    "input_shape": self.config.warmup_input_shape,
                }

            try:
                # Security: Apply timeout to prevent resource exhaustion
                result = await asyncio.wait_for(do_warmup(), timeout=warmup_timeout)
                logger.info(f"Model warmed up with {iterations} iterations")
                return result

            except asyncio.TimeoutError:
                logger.error(
                    f"Warmup timed out after {warmup_timeout}s with {iterations} iterations"
                )
                raise HTTPException(
                    status_code=408,
                    detail=f"Warmup timed out after {warmup_timeout} seconds",
                )
            except Exception as e:
                logger.error(f"Warmup error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Warmup failed")

        self._app = app
        return app

    def serve(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """Start the server.

        Args:
            host: Override host from config
            port: Override port from config
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required for serving. " "Install with: pip install uvicorn"
            )

        if self._app is None:
            self._create_app()

        # Security: Warn about ineffective rate limiting in multi-worker setup
        if self.config.workers > 1 and self.config.rate_limit_per_minute > 0:
            logger.warning(
                "SECURITY WARNING: In-memory rate limiting is enabled with %d workers. "
                "Rate limits are NOT shared across workers and will be ineffective. "
                "For production multi-worker deployments, use an external rate limiter "
                "(Redis, nginx, etc.) and set rate_limit_per_minute=0.",
                self.config.workers,
            )

        # Prepare uvicorn config
        uvicorn_config = {
            "host": host or self.config.host,
            "port": port or self.config.port,
            "workers": self.config.workers,
            "timeout_keep_alive": self.config.timeout,
        }

        # Add SSL if configured
        if self.config.ssl_certfile and self.config.ssl_keyfile:
            uvicorn_config["ssl_certfile"] = self.config.ssl_certfile
            uvicorn_config["ssl_keyfile"] = self.config.ssl_keyfile
            logger.info("Starting server with TLS/HTTPS enabled")
        else:
            logger.warning(
                "Server starting without TLS. Consider configuring ssl_certfile "
                "and ssl_keyfile for production use."
            )

        uvicorn.run(self._app, **uvicorn_config)

    @property
    def app(self):
        """Get the FastAPI app (for testing or custom configuration)."""
        if self._app is None:
            self._create_app()
        return self._app


def create_app(
    model,
    preprocess: Optional[Callable] = None,
    postprocess: Optional[Callable] = None,
    **config_kwargs,
):
    """Create a FastAPI app for model serving.

    Convenience function for creating a server app.

    Args:
        model: PyFlame model
        preprocess: Input preprocessing function
        postprocess: Output postprocessing function
        **config_kwargs: ServerConfig arguments

    Returns:
        FastAPI application

    Example:
        >>> app = create_app(model)
        >>> # Use with uvicorn: uvicorn main:app --reload
    """
    config = ServerConfig(**config_kwargs)
    server = ModelServer(model, config, preprocess, postprocess)
    return server.app


def serve(
    model,
    host: str = "127.0.0.1",  # Security: Default to localhost only
    port: int = 8000,
    **kwargs,
):
    """Start a model server.

    Convenience function for serving a model.

    Args:
        model: PyFlame model
        host: Server host (default: 127.0.0.1 for security).
            Use "0.0.0.0" to accept connections from all interfaces.
        port: Server port
        **kwargs: Additional ServerConfig arguments

    Example:
        >>> serve(model, port=8080)
        >>> serve(model, host="0.0.0.0", port=8080)  # All interfaces
    """
    config = ServerConfig(host=host, port=port, **kwargs)
    server = ModelServer(model, config)
    server.serve()


# Simple HTTP server fallback (no FastAPI dependency)
class SimpleModelServer:
    """Simple HTTP server for model inference (no dependencies).

    WARNING: This server is intended for LOCAL DEVELOPMENT ONLY.
    It lacks many security features present in the main ModelServer:
    - No TLS/HTTPS support
    - No authentication
    - Basic rate limiting only
    - Limited input validation

    For production deployments, use ModelServer with FastAPI instead.

    Example:
        >>> server = SimpleModelServer(model, port=8000)
        >>> server.serve()
    """

    # Security: Maximum request size (10 MB)
    MAX_REQUEST_SIZE = 10 * 1024 * 1024
    # Security: Maximum batch size
    MAX_BATCH_SIZE = 32
    # Security: Basic rate limiting (requests per minute per IP)
    RATE_LIMIT_PER_MINUTE = 60

    def __init__(
        self,
        model,
        host: str = "127.0.0.1",  # Security: Default to localhost only
        port: int = 8000,
    ):
        """Initialize simple server.

        Args:
            model: PyFlame model
            host: Server host (default: 127.0.0.1 for security)
            port: Server port

        Warning:
            This server is for development only. Use ModelServer for production.
        """
        self.model = model
        self.host = host
        self.port = port

        if hasattr(self.model, "eval"):
            self.model.eval()

        # Security: Warn if binding to all interfaces
        if host == "0.0.0.0":
            logger.warning(
                "SimpleModelServer binding to 0.0.0.0 exposes the server on all "
                "network interfaces. This server lacks production security features. "
                "Use ModelServer with FastAPI for production deployments."
            )

    def serve(self):
        """Start the server."""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        model = self.model
        max_request_size = self.MAX_REQUEST_SIZE
        max_batch_size = self.MAX_BATCH_SIZE
        rate_limit = self.RATE_LIMIT_PER_MINUTE

        # Security: Simple in-memory rate limiting with thread safety
        rate_limit_state = defaultdict(list)
        rate_limit_lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def _send_security_headers(self):
                """Add security headers to response."""
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Cache-Control", "no-store")

            def _check_rate_limit(self) -> bool:
                """Check if request is rate limited. Returns True if allowed."""
                if rate_limit <= 0:
                    return True
                client_ip = self.client_address[0]
                current_time = time.time()
                with rate_limit_lock:
                    # Clean old entries
                    rate_limit_state[client_ip] = [
                        t for t in rate_limit_state[client_ip] if current_time - t < 60
                    ]
                    if len(rate_limit_state[client_ip]) >= rate_limit:
                        return False
                    rate_limit_state[client_ip].append(current_time)
                    return True

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self._send_security_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "healthy"}).encode())
                else:
                    self.send_response(404)
                    self._send_security_headers()
                    self.end_headers()

            def do_POST(self):
                # Security: Rate limiting
                if not self._check_rate_limit():
                    self.send_response(429)
                    self.send_header("Content-Type", "application/json")
                    self._send_security_headers()
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"error": "Rate limit exceeded"}).encode()
                    )
                    return

                if self.path == "/predict":
                    try:
                        # Security: Validate content length
                        content_length_header = self.headers.get("Content-Length")
                        if not content_length_header:
                            raise ValueError("Missing Content-Length header")

                        content_length = int(content_length_header)
                        if content_length < 0:
                            raise ValueError("Invalid Content-Length: negative value")
                        if content_length > max_request_size:
                            self.send_response(413)
                            self.send_header("Content-Type", "application/json")
                            self._send_security_headers()
                            self.end_headers()
                            self.wfile.write(
                                json.dumps({"error": "Request too large"}).encode()
                            )
                            return

                        body = self.rfile.read(content_length)
                        data = json.loads(body)

                        # Security: Validate input structure
                        if "inputs" not in data:
                            raise ValueError("Missing 'inputs' field")

                        import numpy as np

                        inputs = np.array(data["inputs"], dtype=np.float32)

                        # Security: Validate batch size
                        if len(inputs.shape) > 0 and inputs.shape[0] > max_batch_size:
                            self.send_response(400)
                            self.send_header("Content-Type", "application/json")
                            self._send_security_headers()
                            self.end_headers()
                            self.wfile.write(
                                json.dumps(
                                    {
                                        "error": f"Batch size exceeds maximum ({max_batch_size})"
                                    }
                                ).encode()
                            )
                            return

                        # Security: Check for NaN/Inf
                        if np.any(np.isnan(inputs)) or np.any(np.isinf(inputs)):
                            self.send_response(400)
                            self.send_header("Content-Type", "application/json")
                            self._send_security_headers()
                            self.end_headers()
                            self.wfile.write(
                                json.dumps(
                                    {
                                        "error": "Input contains invalid values (NaN or Inf)"
                                    }
                                ).encode()
                            )
                            return

                        try:
                            import pyflame as pf

                            inputs = pf.tensor(inputs)
                        except Exception:
                            pass

                        start = time.perf_counter()
                        outputs = model(inputs)
                        end = time.perf_counter()

                        if hasattr(outputs, "numpy"):
                            outputs = outputs.numpy()
                        outputs = np.asarray(outputs).tolist()

                        response = {
                            "outputs": outputs,
                            "inference_time_ms": (end - start) * 1000,
                        }

                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self._send_security_headers()
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode())

                    except json.JSONDecodeError:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self._send_security_headers()
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
                    except ValueError as e:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self._send_security_headers()
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": str(e)}).encode())
                    except Exception:
                        # Security: Don't leak internal error details
                        logger.exception("Inference error in SimpleModelServer")
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self._send_security_headers()
                        self.end_headers()
                        self.wfile.write(
                            json.dumps({"error": "Internal server error"}).encode()
                        )
                else:
                    self.send_response(404)
                    self._send_security_headers()
                    self.end_headers()

            def log_message(self, format, *args):
                # Log at DEBUG level instead of suppressing entirely
                # This allows security auditing while not cluttering normal output
                logger.debug(format % args)

        server = HTTPServer((self.host, self.port), Handler)
        print("PyFlame Simple Model Server (DEVELOPMENT ONLY)")
        print(f"Running at http://{self.host}:{self.port}")
        print("WARNING: Use ModelServer with FastAPI for production deployments.")
        server.serve_forever()
