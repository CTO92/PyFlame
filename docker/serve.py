#!/usr/bin/env python3
"""
PyFlame Model Serving Script for Docker deployment.

This script provides a REST API for model inference.
Configure via environment variables:

    MODEL_PATH: Path to model file (default: /models/model.pf)
    HOST: Server host (default: 0.0.0.0)
    PORT: Server port (default: 8000)
    WORKERS: Number of workers (default: 1)
    LOG_LEVEL: Logging level (default: info)
    CORS_ORIGINS: Comma-separated list of allowed origins (default: localhost only)
    API_KEY: Optional API key for authentication
    RATE_LIMIT: Rate limit per IP per minute (default: 0 = disabled)
    MAX_BATCH_SIZE: Maximum batch size (default: 32)
    MAX_INPUT_MB: Maximum input size in MB (default: 100)
    ALLOWED_MODEL_DIRS: Comma-separated allowed model directories (default: /models)
    SSL_CERTFILE: Path to SSL certificate
    SSL_KEYFILE: Path to SSL key
"""

import hmac
import os
import sys
import json
import time
import logging
from typing import List, Optional, Dict, Any
from collections import defaultdict
from pathlib import Path

# Configure logging with security-relevant information
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s",
)
logger = logging.getLogger("pyflame.serve")

# Add pyflame to path if needed
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/pyflame")

# Security configuration from environment
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
API_KEY = os.environ.get("API_KEY")
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "0"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
MAX_INPUT_MB = float(os.environ.get("MAX_INPUT_MB", "100"))
ALLOWED_MODEL_DIRS = os.environ.get("ALLOWED_MODEL_DIRS", "/models").split(",")


def validate_model_path(model_path: str) -> Path:
    """Validate that model path is within allowed directories.

    Prevents path traversal attacks using proper path containment checks.
    """
    try:
        resolved = Path(model_path).resolve()
    except (ValueError, OSError) as e:
        raise ValueError(f"Invalid path: {e}")

    # Security: Check for null bytes which could bypass validation
    if "\x00" in model_path:
        raise ValueError("Invalid path: contains null bytes")

    # Check if path is within allowed directories using relative_to()
    # This is secure against path traversal (e.g., /models_evil won't match /models)
    for allowed_dir in ALLOWED_MODEL_DIRS:
        try:
            allowed_resolved = Path(allowed_dir).resolve()
            # relative_to() raises ValueError if resolved is not under allowed_resolved
            resolved.relative_to(allowed_resolved)
            # Verify it has a valid model extension
            if resolved.suffix.lower() in ('.pf', '.pt', '.pth', '.onnx', '.safetensors'):
                return resolved
            else:
                raise ValueError(
                    f"Invalid model extension '{resolved.suffix}'. "
                    f"Allowed: .pf, .pt, .pth, .onnx, .safetensors"
                )
        except ValueError:
            # Not under this allowed directory, try next
            continue

    raise ValueError(
        f"Model path is not within allowed directories. "
        f"Allowed: {ALLOWED_MODEL_DIRS}"
    )


def create_app():
    """Create FastAPI application with security features."""
    try:
        from fastapi import FastAPI, HTTPException, Request, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.security import APIKeyHeader
        from pydantic import BaseModel
        from starlette.middleware.base import BaseHTTPMiddleware
    except ImportError:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    import numpy as np

    app = FastAPI(
        title="PyFlame Model Server",
        description="REST API for PyFlame model inference",
        version="1.0.0",
    )

    # Rate limiting state
    rate_limit_state = defaultdict(list)

    # Security headers middleware
    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Cache-Control"] = "no-store"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
            return response

    app.add_middleware(SecurityHeadersMiddleware)

    # Request validation middleware for Content-Type and size limits
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
                        content={"detail": "Content-Type must be application/json"}
                    )

                # Security: Check Content-Length before processing
                content_length = request.headers.get("content-length")
                if content_length:
                    try:
                        size_bytes = int(content_length)
                        max_bytes = int(MAX_INPUT_MB * 1024 * 1024)
                        if size_bytes > max_bytes:
                            return JSONResponse(
                                status_code=413,
                                content={
                                    "detail": f"Request body too large ({size_bytes} bytes). "
                                              f"Maximum: {max_bytes} bytes"
                                }
                            )
                    except ValueError:
                        return JSONResponse(
                            status_code=400,
                            content={"detail": "Invalid Content-Length header"}
                        )

            return await call_next(request)

    app.add_middleware(RequestValidationMiddleware)

    # Enable CORS with restricted origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    # API key authentication
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def verify_api_key(api_key: str = Depends(api_key_header)):
        if API_KEY is not None:
            # Security: Use timing-safe comparison to prevent timing attacks
            if api_key is None or not hmac.compare_digest(
                api_key.encode("utf-8"), API_KEY.encode("utf-8")
            ):
                logger.warning("Invalid API key attempt")
                raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return api_key

    # Rate limiting
    async def check_rate_limit(request: Request):
        if RATE_LIMIT <= 0:
            return
        client_ip = request.client.host
        current_time = time.time()
        # Clean old entries
        rate_limit_state[client_ip] = [
            t for t in rate_limit_state[client_ip] if current_time - t < 60
        ]
        if len(rate_limit_state[client_ip]) >= RATE_LIMIT:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        rate_limit_state[client_ip].append(current_time)

    # Global state
    state = {
        "model": None,
        "engine": None,
        "model_path": os.environ.get("MODEL_PATH", "/models/model.pf"),
        "input_shape": None,
    }

    # Request/Response models
    class PredictRequest(BaseModel):
        inputs: List[List[float]]

    class PredictResponse(BaseModel):
        outputs: List[List[float]]
        inference_time_ms: float

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        model_path: Optional[str] = None

    class StatsResponse(BaseModel):
        total_inferences: int
        average_time_ms: float
        throughput_per_second: float

    @app.on_event("startup")
    async def startup():
        """Load model on startup."""
        model_path = state["model_path"]

        try:
            validated_path = validate_model_path(model_path)
            if not validated_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                logger.info("Server starting without model. Use /v1/load endpoint to load model.")
                return
            await load_model_internal(str(validated_path))
        except ValueError as e:
            logger.error(f"Invalid model path: {e}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    async def load_model_internal(path: str):
        """Internal function to load model."""
        logger.info(f"Loading model from: {path}")

        try:
            import pyflame as pf
            from pyflame.serving import InferenceEngine

            # Load model state
            model_state = pf.load(path)

            # Create simple inference wrapper
            class ModelWrapper:
                def __init__(self, state_dict):
                    self.state_dict = state_dict

                def __call__(self, x):
                    # Placeholder - actual implementation depends on model
                    return x

                def eval(self):
                    pass

            state["model"] = ModelWrapper(model_state)
            state["engine"] = InferenceEngine(state["model"])

            # Warmup
            dummy = pf.randn([1, 10])
            state["engine"].warmup(dummy, num_iterations=5)

            logger.info("Model loaded successfully")

        except ImportError:
            # Fallback without full pyflame
            logger.warning("PyFlame not fully available, using fallback model")

            class FallbackModel:
                def __call__(self, x):
                    return np.asarray(x)

                def eval(self):
                    pass

            state["model"] = FallbackModel()
            state["engine"] = None

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=state["model"] is not None,
            model_path=state["model_path"] if state["model"] else None,
        )

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(
        request: PredictRequest,
        req: Request,
        _: str = Depends(verify_api_key),
        __: None = Depends(check_rate_limit),
    ):
        """Run model prediction with input validation."""
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Input validation
            if not request.inputs:
                raise HTTPException(status_code=400, detail="Empty inputs")

            if len(request.inputs) > MAX_BATCH_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch size {len(request.inputs)} exceeds maximum {MAX_BATCH_SIZE}"
                )

            inputs = np.array(request.inputs, dtype=np.float32)

            # Check input size
            input_size_mb = inputs.nbytes / (1024 * 1024)
            if input_size_mb > MAX_INPUT_MB:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input size {input_size_mb:.1f}MB exceeds maximum {MAX_INPUT_MB}MB"
                )

            # Check for invalid values
            if np.any(np.isnan(inputs)) or np.any(np.isinf(inputs)):
                raise HTTPException(status_code=400, detail="Invalid values (NaN or Inf)")

            start = time.perf_counter()

            if state["engine"]:
                try:
                    import pyflame as pf
                    inputs = pf.tensor(inputs)
                except ImportError:
                    pass
                outputs = state["engine"].infer(inputs)
            else:
                outputs = state["model"](inputs)

            end = time.perf_counter()

            if hasattr(outputs, "numpy"):
                outputs = outputs.numpy()
            outputs = np.asarray(outputs).tolist()

            logger.debug(f"Prediction from {req.client.host}: {input_size_mb:.2f}MB")

            return PredictResponse(
                outputs=outputs if isinstance(outputs[0], list) else [outputs],
                inference_time_ms=(end - start) * 1000,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction error from {req.client.host}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Inference failed")

    @app.get("/v1/stats", response_model=StatsResponse)
    async def stats():
        """Get inference statistics."""
        if state["engine"] is None:
            return StatsResponse(
                total_inferences=0,
                average_time_ms=0.0,
                throughput_per_second=0.0,
            )

        s = state["engine"].stats
        return StatsResponse(
            total_inferences=s.total_inferences,
            average_time_ms=s.average_time_ms,
            throughput_per_second=s.throughput,
        )

    @app.post("/v1/load")
    async def load_model(
        model_path: str,
        _: str = Depends(verify_api_key),
    ):
        """Load a model from path (restricted to allowed directories)."""
        try:
            validated_path = validate_model_path(model_path)
        except ValueError as e:
            logger.warning(f"Rejected model load from invalid path: {model_path}")
            raise HTTPException(status_code=403, detail=str(e))

        if not validated_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        try:
            await load_model_internal(str(validated_path))
            state["model_path"] = str(validated_path)
            logger.info(f"Model loaded: {validated_path}")
            return {"status": "loaded", "model_path": str(validated_path)}
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to load model")

    @app.post("/v1/warmup")
    async def warmup(
        iterations: int = 10,
        _: str = Depends(verify_api_key),
    ):
        """Warmup the model."""
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Limit iterations to prevent DoS
        if iterations < 1 or iterations > 100:
            raise HTTPException(
                status_code=400,
                detail="Iterations must be between 1 and 100"
            )

        if state["engine"]:
            try:
                import pyflame as pf
                dummy = pf.randn([1, 10])
            except ImportError:
                dummy = np.random.randn(1, 10).astype(np.float32)

            state["engine"].warmup(dummy, num_iterations=iterations)

        logger.info(f"Model warmed up with {iterations} iterations")
        return {"status": "warmed up", "iterations": iterations}

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "PyFlame Model Server",
            "version": "1.0.0",
            "endpoints": {
                "health": "/v1/health",
                "predict": "/v1/predict",
                "stats": "/v1/stats",
                "warmup": "/v1/warmup",
                "load": "/v1/load",
            },
        }

    return app


def main():
    """Main entry point."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    ssl_certfile = os.environ.get("SSL_CERTFILE")
    ssl_keyfile = os.environ.get("SSL_KEYFILE")

    logger.info(f"Starting PyFlame Model Server on {host}:{port}")
    if API_KEY:
        logger.info("API key authentication enabled")
    if RATE_LIMIT > 0:
        logger.info(f"Rate limiting enabled: {RATE_LIMIT} requests/minute")

    app = create_app()

    uvicorn_config = {
        "host": host,
        "port": port,
        "workers": workers,
        "log_level": log_level,
    }

    if ssl_certfile and ssl_keyfile:
        uvicorn_config["ssl_certfile"] = ssl_certfile
        uvicorn_config["ssl_keyfile"] = ssl_keyfile
        logger.info("TLS/HTTPS enabled")
    else:
        logger.warning("TLS/HTTPS not configured. Consider enabling for production.")

    uvicorn.run(app, **uvicorn_config)


if __name__ == "__main__":
    main()
