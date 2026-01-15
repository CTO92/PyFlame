"""
Model serving server for PyFlame.

Provides HTTP/REST API for model inference.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for model server.

    Attributes:
        host: Server host
        port: Server port
        workers: Number of worker processes
        timeout: Request timeout in seconds
        max_batch_size: Maximum batch size
        max_input_size_mb: Maximum input size in megabytes
        enable_cors: Enable CORS
        cors_origins: List of allowed CORS origins (use specific origins in production)
        api_prefix: API route prefix
        api_key: Optional API key for authentication
        rate_limit_per_minute: Rate limit per client IP per minute (0 = disabled)
        ssl_certfile: Path to SSL certificate file
        ssl_keyfile: Path to SSL key file
    """

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout: int = 60
    max_batch_size: int = 32
    max_input_size_mb: float = 100.0
    enable_cors: bool = True
    cors_origins: List[str] = None  # None defaults to localhost only
    api_prefix: str = "/v1"
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 0
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None


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

        # Rate limiting state
        rate_limit_state = defaultdict(list)

        # Security middleware for headers
        class SecurityHeadersMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                response = await call_next(request)
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["X-XSS-Protection"] = "1; mode=block"
                response.headers["Cache-Control"] = "no-store"
                return response

        app.add_middleware(SecurityHeadersMiddleware)

        # Enable CORS if configured with restricted origins
        if self.config.enable_cors:
            # Default to localhost only if no origins specified
            allowed_origins = self.config.cors_origins or [
                "http://localhost:3000",
                "http://localhost:8080",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080",
            ]
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
                if api_key != self.config.api_key:
                    raise HTTPException(
                        status_code=401, detail="Invalid or missing API key"
                    )
            return api_key

        # Rate limiting dependency
        async def check_rate_limit(request: Request):
            if self.config.rate_limit_per_minute <= 0:
                return
            client_ip = request.client.host
            current_time = time.time()
            # Clean old entries
            rate_limit_state[client_ip] = [
                t for t in rate_limit_state[client_ip] if current_time - t < 60
            ]
            if len(rate_limit_state[client_ip]) >= self.config.rate_limit_per_minute:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                raise HTTPException(
                    status_code=429, detail="Rate limit exceeded. Try again later."
                )
            rate_limit_state[client_ip].append(current_time)

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
            __: None = Depends(check_rate_limit),
        ):
            """Run model prediction with input validation."""
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

                return PredictResponse(
                    outputs=outputs if isinstance(outputs[0], list) else [outputs],
                    inference_time_ms=inference_time,
                )

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
            try:
                import numpy as np

                # Create dummy input (assumes 2D input, adjust as needed)
                dummy_input = np.random.randn(1, 10).astype(np.float32)

                try:
                    import pyflame as pf

                    dummy_input = pf.tensor(dummy_input)
                except Exception:
                    pass

                self._engine.warmup(dummy_input, num_iterations=iterations)
                logger.info(f"Model warmed up with {iterations} iterations")
                return {"status": "warmed up", "iterations": iterations}

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
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs,
):
    """Start a model server.

    Convenience function for serving a model.

    Args:
        model: PyFlame model
        host: Server host
        port: Server port
        **kwargs: Additional ServerConfig arguments

    Example:
        >>> serve(model, port=8080)
    """
    config = ServerConfig(host=host, port=port, **kwargs)
    server = ModelServer(model, config)
    server.serve()


# Simple HTTP server fallback (no FastAPI dependency)
class SimpleModelServer:
    """Simple HTTP server for model inference (no dependencies).

    Use this when FastAPI is not available.

    Example:
        >>> server = SimpleModelServer(model, port=8000)
        >>> server.serve()
    """

    def __init__(
        self,
        model,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """Initialize simple server.

        Args:
            model: PyFlame model
            host: Server host
            port: Server port
        """
        self.model = model
        self.host = host
        self.port = port

        if hasattr(self.model, "eval"):
            self.model.eval()

    def serve(self):
        """Start the server."""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        model = self.model

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "healthy"}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/predict":
                    try:
                        content_length = int(self.headers["Content-Length"])
                        body = self.rfile.read(content_length)
                        data = json.loads(body)

                        import numpy as np

                        inputs = np.array(data["inputs"], dtype=np.float32)

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
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode())

                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": str(e)}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress logs

        server = HTTPServer((self.host, self.port), Handler)
        print(f"PyFlame Model Server running at http://{self.host}:{self.port}")
        server.serve_forever()
