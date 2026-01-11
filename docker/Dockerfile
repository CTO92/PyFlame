# PyFlame Docker Image
# Multi-stage build for optimized image size

# =============================================================================
# Build Stage
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Install Python build dependencies
RUN pip install --no-cache-dir \
    scikit-build-core \
    pybind11 \
    numpy \
    wheel

# Build wheel
RUN pip wheel --no-deps --wheel-dir /wheels . || \
    (pip wheel --no-deps --wheel-dir /wheels numpy && \
     cp -r python/pyflame /wheels/)

# =============================================================================
# Runtime Stage
# =============================================================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel from builder
COPY --from=builder /wheels/*.whl /wheels/

# Install PyFlame
RUN pip install --no-cache-dir /wheels/*.whl || \
    (pip install --no-cache-dir numpy && \
     mkdir -p /app/pyflame)

# Copy Python source as fallback
COPY --from=builder /app/python/pyflame /app/pyflame

# Add to Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Install serving dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    numpy

# Copy serving script
COPY docker/serve.py .

# Create non-root user
RUN useradd -m -u 1000 pyflame && \
    chown -R pyflame:pyflame /app
USER pyflame

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Default command
CMD ["python", "serve.py"]

# =============================================================================
# Development Stage
# =============================================================================
FROM runtime as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy \
    ipython \
    jupyter

USER pyflame

# Override command for development
CMD ["bash"]

# =============================================================================
# Jupyter Stage
# =============================================================================
FROM runtime as jupyter

USER root

# Install Jupyter dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    matplotlib \
    pandas \
    seaborn

# Create notebooks directory
RUN mkdir -p /app/notebooks && chown pyflame:pyflame /app/notebooks

USER pyflame

WORKDIR /app/notebooks

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
