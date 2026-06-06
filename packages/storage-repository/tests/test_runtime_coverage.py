"""Coverage for the traced repository wrappers and RepositoryRuntime accessors
that the conformance suite exercises only through the agents repo.

Drives every traced repo (get/save/delete/list, plus memory save_embedding/
vec_search and audit append/list) through RepositoryRuntime, then covers the
`except RepositoryFailure` re-raise path (with on_error callback) and the
migrate/close delegation."""

from __future__ import annotations

from typing import Any

import pytest
from storage_repository import (
    AgentFilter,
    AuditLogEntryFilter,
    AuditLogEntryRecord,
    ChannelFilter,
    EnvironmentFilter,
    MemoryFilter,
    MemoryVecSearchFilter,
    MemoryVecSearchResult,
    MessageFilter,
    RepositoryFailure,
    RepositoryOptions,
    RepositoryRuntime,
    SessionFilter,
    SkillFilter,
    ThreadFilter,
    ToolCallFilter,
    UserProfileFilter,
    VaultFilter,
    WebhookFilter,
)
from storage_repository._runtime import RepositoryDriver

from .conftest import CapturingAuditLog
from .test_conformance import (
    make_agent,
    make_channel,
    make_environment,
    make_memory_entry,
    make_message,
    make_session,
    make_skill,
    make_thread,
    make_tool_call,
    make_user_profile,
    make_vault_entry,
    make_webhook,
)


class _StubRepo:
    """A get/save/delete/list repo that echoes a single canned entity."""

    def __init__(self, entity: Any) -> None:
        self._entity = entity

    async def get(self, _id: str) -> Any:
        return self._entity

    async def save(self, _entity: Any) -> None:
        return None

    async def delete(self, _id: str) -> None:
        return None

    async def list(self, _filter: Any) -> list[Any]:
        return [self._entity]


class _StubMemoryRepo(_StubRepo):
    async def save_embedding(self, _id: str, _embedding: bytes) -> None:
        return None

    async def vec_search(self, _filter: Any) -> list[MemoryVecSearchResult]:
        return [MemoryVecSearchResult(entry=self._entity, distance=0.5)]


class _StubAuditRepo:
    def __init__(self, record: AuditLogEntryRecord) -> None:
        self._record = record

    async def append(self, _record: AuditLogEntryRecord) -> None:
        return None

    async def list(self, _filter: AuditLogEntryFilter) -> list[AuditLogEntryRecord]:
        return [self._record]


def _audit_record() -> AuditLogEntryRecord:
    return AuditLogEntryRecord(
        id="al1",
        level="info",
        event="agent.saved",
        entity_type="agent",
        entity_id="a1",
        operation="save",
        timestamp="2026-01-01T00:00:00Z",
        detail={"k": "v"},
        signature="sig",
    )


class _FullStubDriver(RepositoryDriver):
    def __init__(self) -> None:
        self._agents = _StubRepo(make_agent())
        self._sessions = _StubRepo(make_session())
        self._threads = _StubRepo(make_thread())
        self._messages = _StubRepo(make_message())
        self._tool_calls = _StubRepo(make_tool_call())
        self._skills = _StubRepo(make_skill())
        self._environments = _StubRepo(make_environment())
        self._memory = _StubMemoryRepo(make_memory_entry())
        self._vault = _StubRepo(make_vault_entry())
        self._user_profiles = _StubRepo(make_user_profile())
        self._channels = _StubRepo(make_channel())
        self._webhooks = _StubRepo(make_webhook())
        self._audit = _StubAuditRepo(_audit_record())
        self.migrated = False
        self.closed = False

    @property
    def agents(self) -> Any:
        return self._agents

    @property
    def sessions(self) -> Any:
        return self._sessions

    @property
    def threads(self) -> Any:
        return self._threads

    @property
    def messages(self) -> Any:
        return self._messages

    @property
    def tool_calls(self) -> Any:
        return self._tool_calls

    @property
    def skills(self) -> Any:
        return self._skills

    @property
    def environments(self) -> Any:
        return self._environments

    @property
    def memory(self) -> Any:
        return self._memory

    @property
    def vault(self) -> Any:
        return self._vault

    @property
    def user_profiles(self) -> Any:
        return self._user_profiles

    @property
    def channels(self) -> Any:
        return self._channels

    @property
    def webhooks(self) -> Any:
        return self._webhooks

    @property
    def audit_log_entries(self) -> Any:
        return self._audit

    async def migrate(self) -> None:
        self.migrated = True

    async def close(self) -> None:
        self.closed = True


# repo attr, make_fn, filter
_CASES = [
    ("agents", make_agent, AgentFilter()),
    ("sessions", make_session, SessionFilter()),
    ("threads", make_thread, ThreadFilter()),
    ("messages", make_message, MessageFilter()),
    ("tool_calls", make_tool_call, ToolCallFilter()),
    ("skills", make_skill, SkillFilter()),
    ("environments", make_environment, EnvironmentFilter()),
    ("vault", make_vault_entry, VaultFilter()),
    ("user_profiles", make_user_profile, UserProfileFilter()),
    ("channels", make_channel, ChannelFilter()),
    ("webhooks", make_webhook, WebhookFilter()),
]


@pytest.fixture()
def runtime(mock_span: Any, audit_log: CapturingAuditLog) -> RepositoryRuntime:
    return RepositoryRuntime(_FullStubDriver(), RepositoryOptions(audit_log=audit_log))


@pytest.mark.parametrize(("attr", "make_fn", "filter_obj"), _CASES)
async def test_traced_repo_crud(
    runtime: RepositoryRuntime, attr: str, make_fn: Any, filter_obj: Any
) -> None:
    repo = getattr(runtime, attr)
    entity = make_fn()
    fetched = await repo.get(entity.id)
    assert fetched is not None and fetched.id == entity.id
    await repo.save(entity)
    await repo.delete(entity.id)
    listed = await repo.list(filter_obj)
    assert [e.id for e in listed] == [entity.id]


async def test_traced_memory_repo(runtime: RepositoryRuntime) -> None:
    entry = make_memory_entry()
    fetched = await runtime.memory.get(entry.id)
    assert fetched is not None and fetched.id == entry.id
    await runtime.memory.save(entry)
    await runtime.memory.delete(entry.id)
    assert [e.id for e in await runtime.memory.list(MemoryFilter())] == [entry.id]
    await runtime.memory.save_embedding(entry.id, b"\x00\x01")
    results = await runtime.memory.vec_search(MemoryVecSearchFilter(embedding=b"\x00\x01", limit=5))
    assert len(results) == 1 and results[0].distance == 0.5


async def test_traced_audit_repo(runtime: RepositoryRuntime) -> None:
    record = _audit_record()
    await runtime.audit_log_entries.append(record)
    assert await runtime.audit_log_entries.list(AuditLogEntryFilter()) == [record]


async def test_repository_failure_reraised_and_on_error_called(
    mock_span: Any, audit_log: CapturingAuditLog
) -> None:
    failure = RepositoryFailure(
        code="ALREADY_FAILED",
        message="precomputed",
        entity_type="agent",
        entity_id="a1",
        operation="save",
        timestamp="2026-01-01T00:00:00Z",
        cause=None,
    )

    class _RaisingAgentRepo(_StubRepo):
        async def save(self, _entity: Any) -> None:
            raise failure

    driver = _FullStubDriver()
    driver._agents = _RaisingAgentRepo(make_agent())
    captured: list[RepositoryFailure] = []
    runtime = RepositoryRuntime(
        driver,
        RepositoryOptions(audit_log=audit_log, on_error=captured.append),
    )

    with pytest.raises(RepositoryFailure) as exc_info:
        await runtime.agents.save(make_agent())
    assert exc_info.value is failure
    assert captured == [failure]
    assert audit_log.entries[0].event == "repo.agent.save.failed"


async def test_repository_failure_reraised_without_on_error(
    mock_span: Any, audit_log: CapturingAuditLog
) -> None:
    failure = RepositoryFailure(
        code="ALREADY_FAILED",
        message="precomputed",
        entity_type="agent",
        entity_id="a1",
        operation="save",
        timestamp="2026-01-01T00:00:00Z",
        cause=None,
    )

    class _RaisingAgentRepo(_StubRepo):
        async def save(self, _entity: Any) -> None:
            raise failure

    driver = _FullStubDriver()
    driver._agents = _RaisingAgentRepo(make_agent())
    runtime = RepositoryRuntime(driver, RepositoryOptions(audit_log=audit_log))

    with pytest.raises(RepositoryFailure) as exc_info:
        await runtime.agents.save(make_agent())
    assert exc_info.value is failure


async def test_migrate_and_close_delegate(mock_span: Any, audit_log: CapturingAuditLog) -> None:
    driver = _FullStubDriver()
    runtime = RepositoryRuntime(driver, RepositoryOptions(audit_log=audit_log))
    await runtime.migrate()
    await runtime.close()
    assert driver.migrated is True
    assert driver.closed is True
