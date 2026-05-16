from __future__ import annotations

from opentelemetry import trace

from ._version import CAPABILITIES_SDK_VERSION

_TRACER_NAME = "meridian.sdk-capabilities"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, CAPABILITIES_SDK_VERSION)
