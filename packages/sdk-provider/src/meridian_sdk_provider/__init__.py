"""Meridian Model Provider SDK.

Public surface for provider authors and router consumers::

    from meridian_sdk_provider import (
        ModelProvider,
        ProviderCapabilities,
        ModelCallOpts,
        ModelEvent,
        ModelRouter,
        ModelRoutingPolicy,
        ModelRoutingRule,
        FallbackRule,
        AuditLog,
        AuditLogEntry,
    )
"""

from ._version import SDK_PROVIDER_VERSION
from .audit import AuditLog, AuditLogEntry, NoopAuditLog
from .fake import FakeModelAdapter, write_model_fixture
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider
from .errors import (
    NoProviderFoundError,
    ProviderCallError,
    ProviderError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    RoutingError,
)
from .protocol import ModelCapabilities, ModelEntry, ModelProvider, ProviderCapabilities
from .router import (
    FallbackRule,
    ModelRouter,
    ModelRoutingPolicy,
    ModelRoutingRule,
    RoutingCondition,
    TokenRange,
)
from .telemetry import get_tracer, record_invocation_event, record_provider_failure
from .types import (
    CacheControl,
    ContentBlock,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    ModelCountReq,
    ModelEvent,
    TextBlock,
    TextDeltaEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    TokenCount,
    ToolDefinition,
    ToolInputDeltaEvent,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseStartEvent,
)

__version__ = SDK_PROVIDER_VERSION

__all__ = [
    # Protocol + capabilities
    "ModelProvider",
    "ProviderCapabilities",
    "ModelCapabilities",
    "ModelEntry",
    # Router
    "ModelRouter",
    "ModelRoutingPolicy",
    "ModelRoutingRule",
    "RoutingCondition",
    "TokenRange",
    "FallbackRule",
    # Core types
    "ModelCallOpts",
    "ModelCountReq",
    "TokenCount",
    "ModelEvent",
    "MessageStartEvent",
    "TextDeltaEvent",
    "ThinkingDeltaEvent",
    "ToolUseStartEvent",
    "ToolInputDeltaEvent",
    "MessageDeltaEvent",
    "MessageStopEvent",
    # Message / content types
    "Message",
    "ContentBlock",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ThinkingBlock",
    "CacheControl",
    "ToolDefinition",
    # Audit
    "AuditLog",
    "AuditLogEntry",
    "NoopAuditLog",
    # Fake / testing
    "FakeModelAdapter",
    "write_model_fixture",
    # Providers
    "OllamaProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    # Telemetry helpers
    "get_tracer",
    "record_invocation_event",
    "record_provider_failure",
    # Errors
    "ProviderError",
    "ProviderCallError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderServerError",
    "NoProviderFoundError",
    "RoutingError",
    # Version
    "__version__",
    "SDK_PROVIDER_VERSION",
]
