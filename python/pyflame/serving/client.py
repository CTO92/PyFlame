"""
Model client for PyFlame serving.

Provides HTTP client for calling PyFlame model servers.
"""

import json
import logging
import ssl
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for model client.

    Attributes:
        base_url: Server base URL (defaults to HTTPS)
        timeout: Request timeout in seconds
        api_prefix: API route prefix
        headers: Additional headers
        verify_ssl: Whether to verify SSL certificates (default: True)
        allow_insecure_http: Explicitly allow HTTP connections (default: False)
    """

    base_url: str = "https://localhost:8000"
    timeout: int = 30
    api_prefix: str = "/v1"
    headers: Optional[Dict[str, str]] = None
    verify_ssl: bool = True
    allow_insecure_http: bool = False


class ModelClient:
    """HTTP client for PyFlame model servers.

    Provides a simple interface for calling remote model servers.

    Example:
        >>> client = ModelClient("http://localhost:8000")
        >>> output = client.predict([[1.0, 2.0, 3.0]])
        >>> print(output)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        config: Optional[ClientConfig] = None,
    ):
        """Initialize client.

        Args:
            base_url: Server base URL
            config: Client configuration
        """
        if config:
            self.config = config
        else:
            self.config = ClientConfig(base_url=base_url or "https://localhost:8000")

        # Security: Warn about insecure HTTP connections
        self._validate_url_security()

        self._session = None
        self._ssl_context = self._create_ssl_context()

    def _validate_url_security(self) -> None:
        """Validate URL security and warn about insecure connections."""
        url = self.config.base_url.lower()

        if url.startswith("http://"):
            if not self.config.allow_insecure_http:
                warnings.warn(
                    "Using insecure HTTP connection. Set allow_insecure_http=True "
                    "to suppress this warning, or use HTTPS for secure connections.",
                    UserWarning,
                    stacklevel=3,
                )
            logger.warning(
                "SECURITY: Using insecure HTTP connection to %s. "
                "Consider using HTTPS for production.",
                self.config.base_url,
            )

    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for secure connections.

        Returns:
            SSL context configured for secure connections, or None for HTTP.
        """
        if self.config.base_url.lower().startswith("http://"):
            return None

        if self.config.verify_ssl:
            # Create a secure SSL context with certificate verification
            context = ssl.create_default_context()
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
        else:
            # Allow unverified SSL (not recommended for production)
            warnings.warn(
                "SSL certificate verification is disabled. "
                "This is insecure and not recommended for production.",
                UserWarning,
                stacklevel=3,
            )
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        return context

    def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            try:
                import requests

                self._session = requests.Session()
                # Configure SSL verification
                self._session.verify = self.config.verify_ssl
                if self.config.headers:
                    self._session.headers.update(self.config.headers)
            except ImportError:
                self._session = None
        return self._session

    def _build_url(self, endpoint: str) -> str:
        """Build full URL for endpoint.

        Args:
            endpoint: API endpoint

        Returns:
            Full URL
        """
        base = self.config.base_url.rstrip("/")
        prefix = self.config.api_prefix
        return f"{base}{prefix}{endpoint}"

    def predict(
        self,
        inputs: Union[List[List[float]], Any],
        return_time: bool = False,
    ) -> Union[List[List[float]], tuple]:
        """Run prediction on remote model.

        Args:
            inputs: Input data (list of lists or numpy array)
            return_time: Return inference time

        Returns:
            Model outputs, optionally with inference time

        Example:
            >>> outputs = client.predict([[1.0, 2.0, 3.0]])
            >>> outputs, time_ms = client.predict([[1.0, 2.0, 3.0]], return_time=True)
        """
        # Convert numpy arrays to lists
        if hasattr(inputs, "tolist"):
            inputs = inputs.tolist()

        session = self._get_session()

        if session:
            # Use requests library
            response = session.post(
                self._build_url("/predict"),
                json={"inputs": inputs},
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
        else:
            # Fallback to urllib
            data = self._request_urllib(
                "POST",
                "/predict",
                {"inputs": inputs},
            )

        outputs = data["outputs"]
        inference_time = data.get("inference_time_ms", 0.0)

        if return_time:
            return outputs, inference_time
        return outputs

    def batch_predict(
        self,
        inputs: List[List[List[float]]],
    ) -> List[List[List[float]]]:
        """Run batch prediction on remote model.

        Args:
            inputs: List of input batches

        Returns:
            List of output batches
        """
        results = []
        for batch in inputs:
            output = self.predict(batch)
            results.append(output)
        return results

    def health(self) -> Dict[str, Any]:
        """Check server health.

        Returns:
            Health status dictionary

        Example:
            >>> status = client.health()
            >>> print(status["status"])  # "healthy"
        """
        session = self._get_session()

        if session:
            response = session.get(
                self._build_url("/health"),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()
        else:
            return self._request_urllib("GET", "/health")

    def stats(self) -> Dict[str, Any]:
        """Get server statistics.

        Returns:
            Statistics dictionary
        """
        session = self._get_session()

        if session:
            response = session.get(
                self._build_url("/stats"),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()
        else:
            return self._request_urllib("GET", "/stats")

    def warmup(self, iterations: int = 10) -> Dict[str, Any]:
        """Warmup the remote model.

        Args:
            iterations: Number of warmup iterations

        Returns:
            Warmup status
        """
        session = self._get_session()

        if session:
            response = session.post(
                self._build_url("/warmup"),
                params={"iterations": iterations},
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()
        else:
            return self._request_urllib(
                "POST",
                f"/warmup?iterations={iterations}",
            )

    def _request_urllib(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make request using urllib (fallback).

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body

        Returns:
            Response data
        """
        import urllib.error
        import urllib.request

        url = self._build_url(endpoint)

        if data:
            body = json.dumps(data).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=body,
                method=method,
                headers={"Content-Type": "application/json"},
            )
        else:
            request = urllib.request.Request(url, method=method)

        try:
            # Security: Use SSL context for HTTPS connections
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout,
                context=self._ssl_context,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error: {e.reason}")

    def close(self):
        """Close the client session."""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def connect(
    url: str = "https://localhost:8000",
    **kwargs,
) -> ModelClient:
    """Connect to a PyFlame model server.

    Convenience function for creating a client.

    Args:
        url: Server URL (defaults to HTTPS for security)
        **kwargs: Additional ClientConfig arguments including:
            - verify_ssl: Whether to verify SSL certificates (default: True)
            - allow_insecure_http: Allow HTTP connections (default: False)

    Returns:
        ModelClient instance

    Example:
        >>> client = connect("https://localhost:8000")
        >>> output = client.predict([[1.0, 2.0, 3.0]])

        # For local development with HTTP (not recommended for production):
        >>> client = connect("http://localhost:8000", allow_insecure_http=True)
    """
    config = ClientConfig(base_url=url, **kwargs)
    return ModelClient(config=config)
