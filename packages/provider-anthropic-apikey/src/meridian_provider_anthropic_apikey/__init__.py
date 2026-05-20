"""Meridian AnthropicApiKeyProvider — Mode 1 Anthropic adapter.

Uses the raw anthropic Python SDK with an API key resolved from Vault.
Connects directly to api.anthropic.com with per-token billing and the
full Anthropic feature surface (streaming, tool-use, vision, extended
thinking, prompt caching, token counting).
"""

from ._version import ANTHROPIC_APIKEY_PROVIDER_VERSION
from .provider import AnthropicApiKeyProvider

__version__ = ANTHROPIC_APIKEY_PROVIDER_VERSION

__all__ = [
    "AnthropicApiKeyProvider",
    "ANTHROPIC_APIKEY_PROVIDER_VERSION",
    "__version__",
]
