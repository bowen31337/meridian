"""
ULID conformance suite.

Covers MonotonicUlidGenerator, generate_ulid, IdPrefix, and UlidRuntime:

  MonotonicUlidGenerator:
    - generate() returns a 26-character string.
    - All characters are from the Crockford base32 alphabet.
    - Successive calls within the same millisecond produce strictly increasing values.
    - Calls spanning different milliseconds produce values where the later one
      compares greater lexicographically (timestamp portion dominates).
    - Multiple generator instances maintain independent state.

  IdPrefix:
    - All 14 typed prefixes are present with correct string values.

  UlidRuntime.generate — success:
    - Returns "<prefix>_<26-char ULID>" for every IdPrefix value.
    - Span name is "ulid.generate".
    - Span carries "ulid.prefix" attribute.
    - "ulid.invocation" event attached with operation="generate".
    - No audit entries on success.
    - Span ended on success.

  UlidRuntime.generate — generator raises UlidFailure:
    - UlidFailure is re-raised unchanged.
    - Audit entry written with level="error", event="ulid.generate.failed".
    - Span marked ERROR.
    - Span ended on failure.

  UlidRuntime.generate — generator raises unexpected exception:
    - Wrapped as UlidFailure with code="ULID_GENERATE_FAILED".
    - Original exception preserved as cause.
    - Audit entry written with level="error", event="ulid.generate.failed".
    - Span marked ERROR.
    - "ulid.error" event attached.
    - Exception recorded on span.
    - on_error callback invoked.
    - Span ended on failure.
"""

from __future__ import annotations

from unittest.mock import patch

from opentelemetry.trace import StatusCode
import pytest
from system_ulid import (
    AuditLogEntry,
    IdPrefix,
    MonotonicUlidGenerator,
    UlidFailure,
    UlidOptions,
    UlidRuntime,
    generate_ulid,
)
from system_ulid._generator import _ENCODING

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ULID_ALPHABET = set(_ENCODING)


def make_options(
    audit: CapturingAuditLog,
    errors: list[UlidFailure] | None = None,
) -> UlidOptions:
    return UlidOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


def make_runtime(generator=None) -> UlidRuntime:
    return UlidRuntime(generator=generator)


# ---------------------------------------------------------------------------
# MonotonicUlidGenerator
# ---------------------------------------------------------------------------


class TestMonotonicUlidGenerator:
    def test_returns_26_chars(self) -> None:
        gen = MonotonicUlidGenerator()
        assert len(gen.generate()) == 26

    def test_all_chars_in_crockford_alphabet(self) -> None:
        gen = MonotonicUlidGenerator()
        ulid = gen.generate()
        assert all(c in _ULID_ALPHABET for c in ulid)

    def test_monotonic_same_millisecond(self) -> None:
        gen = MonotonicUlidGenerator()
        with patch("system_ulid._generator._time_ms", return_value=1_700_000_000_000):
            u1 = gen.generate()
            u2 = gen.generate()
            u3 = gen.generate()
        assert u1 < u2 < u3

    def test_monotonic_increasing_milliseconds(self) -> None:
        gen = MonotonicUlidGenerator()
        with patch("system_ulid._generator._time_ms", return_value=1_700_000_000_000):
            u1 = gen.generate()
        with patch("system_ulid._generator._time_ms", return_value=1_700_000_000_001):
            u2 = gen.generate()
        assert u1 < u2

    def test_same_ms_increments_random_not_timestamp(self) -> None:
        gen = MonotonicUlidGenerator()
        with patch("system_ulid._generator._time_ms", return_value=1_700_000_000_000):
            u1 = gen.generate()
            u2 = gen.generate()
        # Timestamp portion (first 10 chars) must be identical
        assert u1[:10] == u2[:10]
        # Random portion (last 16 chars) must differ
        assert u1[10:] != u2[10:]

    def test_independent_instances_do_not_share_state(self) -> None:
        gen_a = MonotonicUlidGenerator()
        gen_b = MonotonicUlidGenerator()
        with patch("system_ulid._generator._time_ms", return_value=1_700_000_000_000):
            a1 = gen_a.generate()
            b1 = gen_b.generate()
            a2 = gen_a.generate()
            b2 = gen_b.generate()
        # gen_a's second call increments its own random, not gen_b's
        assert a1 < a2
        # gen_b's sequences are also monotonic
        assert b1 < b2
        # But they started from independent random seeds, so may differ
        assert a1 != b1 or a2 != b2  # at least one differs (random seeds differ)


class TestGenerateUlidFunction:
    def test_returns_26_chars(self) -> None:
        assert len(generate_ulid()) == 26

    def test_all_chars_in_crockford_alphabet(self) -> None:
        ulid = generate_ulid()
        assert all(c in _ULID_ALPHABET for c in ulid)

    def test_successive_calls_are_non_decreasing(self) -> None:
        results = [generate_ulid() for _ in range(100)]
        for a, b in zip(results, results[1:], strict=False):
            assert a <= b


# ---------------------------------------------------------------------------
# IdPrefix
# ---------------------------------------------------------------------------


class TestIdPrefix:
    def test_all_14_prefixes_present(self) -> None:
        expected = {
            "agent",
            "sess",
            "thr",
            "msg",
            "tc",
            "skill",
            "skillver",
            "env",
            "mem",
            "vault",
            "usr",
            "chan",
            "file",
            "wh",
        }
        actual = {p.value for p in IdPrefix}
        assert actual == expected

    def test_agent_value(self) -> None:
        assert IdPrefix.AGENT.value == "agent"

    def test_sess_value(self) -> None:
        assert IdPrefix.SESS.value == "sess"

    def test_thr_value(self) -> None:
        assert IdPrefix.THR.value == "thr"

    def test_msg_value(self) -> None:
        assert IdPrefix.MSG.value == "msg"

    def test_tc_value(self) -> None:
        assert IdPrefix.TC.value == "tc"

    def test_skill_value(self) -> None:
        assert IdPrefix.SKILL.value == "skill"

    def test_skillver_value(self) -> None:
        assert IdPrefix.SKILLVER.value == "skillver"

    def test_env_value(self) -> None:
        assert IdPrefix.ENV.value == "env"

    def test_mem_value(self) -> None:
        assert IdPrefix.MEM.value == "mem"

    def test_vault_value(self) -> None:
        assert IdPrefix.VAULT.value == "vault"

    def test_usr_value(self) -> None:
        assert IdPrefix.USR.value == "usr"

    def test_chan_value(self) -> None:
        assert IdPrefix.CHAN.value == "chan"

    def test_file_value(self) -> None:
        assert IdPrefix.FILE.value == "file"

    def test_wh_value(self) -> None:
        assert IdPrefix.WH.value == "wh"


# ---------------------------------------------------------------------------
# UlidRuntime.generate — success
# ---------------------------------------------------------------------------


class TestGenerateSuccess:
    def test_returns_prefixed_ulid(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = make_runtime()
        result = rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert result.startswith("agent_")
        suffix = result[len("agent_") :]
        assert len(suffix) == 26
        assert all(c in _ULID_ALPHABET for c in suffix)

    def test_all_prefixes_produce_correct_format(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = make_runtime()
        for prefix in IdPrefix:
            result = rt.generate(prefix, options=make_options(audit_log))
            assert result.startswith(f"{prefix.value}_"), f"bad format for {prefix}"
            assert len(result) == len(prefix.value) + 1 + 26

    def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        make_runtime().generate(IdPrefix.SESS, options=make_options(audit_log))
        assert mock_span.name == "ulid.generate"

    def test_span_prefix_attribute(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        make_runtime().generate(IdPrefix.MSG, options=make_options(audit_log))
        assert mock_span.attributes["ulid.prefix"] == "msg"

    def test_invocation_event_attached(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().generate(IdPrefix.AGENT, options=make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "ulid.invocation" in event_names

    def test_invocation_event_operation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().generate(IdPrefix.AGENT, options=make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "ulid.invocation")
        assert inv[1]["operation"] == "generate"

    def test_invocation_event_prefix(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().generate(IdPrefix.THR, options=make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "ulid.invocation")
        assert inv[1]["prefix"] == "thr"

    def test_no_audit_entries_on_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        make_runtime().generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert audit_log.entries == []

    def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        make_runtime().generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert mock_span.ended

    def test_default_options_used_when_none_given(self, mock_span: MockSpan) -> None:
        result = make_runtime().generate(IdPrefix.USR)
        assert result.startswith("usr_")


# ---------------------------------------------------------------------------
# UlidRuntime.generate — generator raises UlidFailure
# ---------------------------------------------------------------------------


class TestGenerateUlidFailure:
    def _make_failure(self) -> UlidFailure:
        return UlidFailure(
            code="ULID_OVERFLOW",
            message="random component overflow",
            prefix="agent",
            timestamp="2024-01-01T00:00:00+00:00",
        )

    def test_re_raises_ulid_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        failure = self._make_failure()
        rt = make_runtime(generator=lambda: (_ for _ in ()).throw(failure))
        with pytest.raises(UlidFailure) as exc_info:
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert exc_info.value.code == "ULID_OVERFLOW"

    def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = make_runtime(generator=lambda: (_ for _ in ()).throw(self._make_failure()))
        with pytest.raises(UlidFailure):
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "ulid.generate.failed"

    def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = make_runtime(generator=lambda: (_ for _ in ()).throw(self._make_failure()))
        with pytest.raises(UlidFailure):
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    def test_span_ended_on_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = make_runtime(generator=lambda: (_ for _ in ()).throw(self._make_failure()))
        with pytest.raises(UlidFailure):
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# UlidRuntime.generate — generator raises unexpected exception
# ---------------------------------------------------------------------------


class TestGenerateStoreRaises:
    def test_wraps_as_generate_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        def boom() -> str:
            raise OSError("entropy device error")

        rt = make_runtime(generator=boom)
        with pytest.raises(UlidFailure) as exc_info:
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert exc_info.value.code == "ULID_GENERATE_FAILED"

    def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = OSError("entropy device error")

        def boom() -> str:
            raise orig

        rt = make_runtime(generator=boom)
        with pytest.raises(UlidFailure) as exc_info:
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert exc_info.value.cause is orig

    def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        def boom() -> str:
            raise RuntimeError("boom")

        rt = make_runtime(generator=boom)
        with pytest.raises(UlidFailure):
            rt.generate(IdPrefix.MEM, options=make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "ulid.generate.failed"
        assert entry.prefix == "mem"

    def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        def boom() -> str:
            raise RuntimeError("boom")

        rt = make_runtime(generator=boom)
        with pytest.raises(UlidFailure):
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    def test_error_event_on_span(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        def boom() -> str:
            raise RuntimeError("boom")

        rt = make_runtime(generator=boom)
        with pytest.raises(UlidFailure):
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "ulid.error" in event_names

    def test_exception_recorded_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = RuntimeError("boom")

        def boom() -> str:
            raise orig

        rt = make_runtime(generator=boom)
        with pytest.raises(UlidFailure):
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert orig in mock_span.recorded_exceptions

    def test_on_error_callback(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        errors: list[UlidFailure] = []

        def boom() -> str:
            raise RuntimeError("boom")

        rt = make_runtime(generator=boom)
        with pytest.raises(UlidFailure):
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "ULID_GENERATE_FAILED"

    def test_span_ended_on_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        def boom() -> str:
            raise RuntimeError("boom")

        rt = make_runtime(generator=boom)
        with pytest.raises(UlidFailure):
            rt.generate(IdPrefix.AGENT, options=make_options(audit_log))
        assert mock_span.ended
