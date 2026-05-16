from __future__ import annotations


class ProviderError(Exception):
    """Base class for all SDK-level provider errors."""


class ProviderCallError(ProviderError):
    """A model call failed at the provider layer."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_name = provider_name
        self.status_code = status_code


class ProviderRateLimitError(ProviderCallError):
    """Provider returned a rate-limit response (HTTP 429 or equivalent)."""


class ProviderTimeoutError(ProviderCallError):
    """Model call exceeded the configured timeout."""


class ProviderServerError(ProviderCallError):
    """Provider returned a 5xx response."""


class NoProviderFoundError(ProviderError):
    """Router could not find a registered provider for the requested model ref."""


class RoutingError(ProviderError):
    """Routing policy is invalid or could not be evaluated."""
