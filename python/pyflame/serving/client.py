"""
Model client for PyFlame serving.

Provides HTTP client for calling PyFlame model servers.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ClientConfig:
    """Configuration for model client.

    Attributes:
        base_url: Server base URL
        timeout: Request timeout in seconds
        api_prefix: API route prefix
        headers: Additional headers
    """

    base_url: str = "http://localhost:8000"
    timeout: int = 30
    api_prefix: str = "/v1"
    headers: Optional[Dict[str, str]] = None


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
            self.config = ClientConfig(base_url=base_url or "http://localhost:8000")

        self._session = None

    def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            try:
                import requests

                self._session = requests.Session()
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
            with urllib.request.urlopen(
                request, timeout=self.config.timeout
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
    url: str = "http://localhost:8000",
    **kwargs,
) -> ModelClient:
    """Connect to a PyFlame model server.

    Convenience function for creating a client.

    Args:
        url: Server URL
        **kwargs: Additional ClientConfig arguments

    Returns:
        ModelClient instance

    Example:
        >>> client = connect("http://localhost:8000")
        >>> output = client.predict([[1.0, 2.0, 3.0]])
    """
    config = ClientConfig(base_url=url, **kwargs)
    return ModelClient(config=config)
