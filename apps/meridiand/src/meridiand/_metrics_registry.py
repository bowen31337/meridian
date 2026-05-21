from __future__ import annotations

from prometheus_client import Counter, Histogram

sessions_total = Counter(
    "meridian_sessions_total",
    "Total number of sessions entering each lifecycle phase",
    ["phase"],
)

session_duration_seconds = Histogram(
    "meridian_session_duration_seconds",
    "Session wall-clock duration in seconds, labelled by terminal result",
    ["result"],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0),
)

tool_calls_total = Counter(
    "meridian_tool_calls_total",
    "Total number of tool calls by tool name, backend, and result",
    ["tool", "backend", "result"],
)

tool_call_duration_seconds = Histogram(
    "meridian_tool_call_duration_seconds",
    "Tool call round-trip duration in seconds (checkpoint-interval approximation)",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
)
