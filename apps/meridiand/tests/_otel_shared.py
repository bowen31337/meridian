"""Shared OTel TracerProvider registered once per process for the test suite."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

otel_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(otel_exporter))
if not isinstance(trace.get_tracer_provider(), TracerProvider):
    trace.set_tracer_provider(_provider)
