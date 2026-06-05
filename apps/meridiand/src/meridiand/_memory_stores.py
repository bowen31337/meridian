from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Literal
import uuid

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from meridian_kb_indexer import Chunk
from meridian_sdk_provider import (
    Message,
    ModelCallOpts,
    ModelRouter,
    TextDeltaEvent,
)
from pydantic import BaseModel
from storage_event_log import EventLogRuntime

from ._kb import KbStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class MemoryStoreCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="memory_store_create_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class MemoryStoreInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="memory_store_invalid_request", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 422


class MemoryStoreNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="memory_store_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


class MemoryStoreQueryError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="memory_store_query_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class MemoryStoreWriteError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="memory_store_write_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class MemoryStoreDialecticError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="memory_store_dialectic_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

MemoryStoreBackend = Literal["sqlite-vec", "pgvector", "http"]
MemoryStoreScope = Literal["global", "user", "agent", "project"]


class MemoryStoreCreateRequest(BaseModel):
    name: str
    backend: MemoryStoreBackend
    scope: MemoryStoreScope
    metadata: dict[str, Any] | None = None


class MemoryStoreQueryRequest(BaseModel):
    query: str
    scope: str | None = None
    limit: int = 10
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    rrf_k: int = 60


class MemoryStoreWriteRequest(BaseModel):
    key: str
    content: str
    scope: str | None = None
    embedder_id: str | None = None
    dialectic: bool = False
    dialectic_top_k: int = 5


def _validate_request(body: MemoryStoreCreateRequest) -> MemoryStoreInvalidRequestError | None:
    if not body.name.strip():
        return MemoryStoreInvalidRequestError(
            message="'name' must not be empty",
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Weighted RRF fusion
# ---------------------------------------------------------------------------


def _weighted_rrf_fuse(
    ranked_lists: list[tuple[list[dict[str, Any]], float]],
    limit: int,
    k: int = 60,
) -> list[dict[str, Any]]:
    def _key(c: dict[str, Any]) -> tuple[str, int, int]:
        return (c["file_path"], c["start_line"], c["end_line"])

    scores: dict[tuple[str, int, int], float] = {}
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}

    for ranked, weight in ranked_lists:
        for rank, chunk in enumerate(ranked, 1):
            key = _key(chunk)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            by_key[key] = chunk

    sorted_keys = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [by_key[kk] for kk in sorted_keys[:limit]]


# ---------------------------------------------------------------------------
# Dialectic classification
# ---------------------------------------------------------------------------

_DialecticLabel = Literal["duplicate", "refinement", "contradiction", "net-new"]

_CLASSIFIER_SYSTEM = """\
You are a memory deduplication classifier. Given a new memory and existing similar \
memories, classify the relationship.

Labels:
- "duplicate": new memory conveys the same information as an existing one (no new value)
- "refinement": new memory updates or adds compatible information to an existing one
- "contradiction": new memory contradicts an existing one
- "net-new": new memory is distinct from all existing ones

Rules:
- Pick at most one existing memory as the primary match (by key)
- For "refinement", provide merged_content that blends both into a single coherent memory
- For all other labels, set merged_content to null
- Respond ONLY with valid JSON, no markdown, no explanation outside the JSON

Response format:
{"label": "duplicate|refinement|contradiction|net-new", "match_key": "<key or null>", \
"merged_content": "<merged text or null>", "explanation": "<one sentence>"}"""


class _DialecticResult:
    __slots__ = ("label", "match_key", "merged_content", "explanation")

    def __init__(
        self,
        label: _DialecticLabel,
        match_key: str | None,
        merged_content: str | None,
        explanation: str,
    ) -> None:
        self.label = label
        self.match_key = match_key
        self.merged_content = merged_content
        self.explanation = explanation


async def _classify_memory(
    model_router: ModelRouter,
    incoming_key: str,
    incoming_content: str,
    candidates: list[dict[str, Any]],
) -> _DialecticResult:
    if not candidates:
        return _DialecticResult(
            label="net-new",
            match_key=None,
            merged_content=None,
            explanation="No existing memories to compare against.",
        )

    candidate_lines = "\n".join(
        f"{i + 1}. [key={c['file_path']!r}]: {c['content']}" for i, c in enumerate(candidates)
    )
    user_text = (
        f"New memory (key: {incoming_key!r}):\n{incoming_content}\n\n"
        f"Existing similar memories:\n{candidate_lines}"
    )

    opts = ModelCallOpts(
        model="memory_classifier",
        messages=[Message(role="user", content=user_text)],
        system=_CLASSIFIER_SYSTEM,
        max_tokens=512,
        temperature=0.0,
        role="memory_classifier",
        skill_id="memory_dialectic_classifier",
    )

    text_parts: list[str] = []
    async for event in model_router.call(opts):
        if isinstance(event, TextDeltaEvent):
            text_parts.append(event.text)

    raw = "".join(text_parts).strip()
    try:
        data = json.loads(raw)
        label: _DialecticLabel = data["label"]
        if label not in ("duplicate", "refinement", "contradiction", "net-new"):
            raise ValueError(f"Unknown label: {label!r}")
        return _DialecticResult(
            label=label,
            match_key=data.get("match_key"),
            merged_content=data.get("merged_content"),
            explanation=data.get("explanation", ""),
        )
    except Exception as exc:
        raise ValueError(f"Classifier returned invalid response: {raw!r}") from exc


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_memory_stores_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    model_router: ModelRouter | None = None,
    event_log: EventLogRuntime | None = None,
) -> APIRouter:
    router = APIRouter()
    stores_dir = storage_root / "memory_stores"

    @router.post("/v1/memory_stores", status_code=201)
    async def create_memory_store(body: MemoryStoreCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        store_id = f"memstore_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "memory_store.create",
            attributes={
                "memory_store.id": store_id,
                "memory_store.name": body.name,
                "memory_store.backend": body.backend,
                "memory_store.scope": body.scope,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="memory_store.create.invocation",
                    code="memory_store_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_request(body)
                if validation_err is not None:
                    raise validation_err

                stores_dir.mkdir(parents=True, exist_ok=True)

                store_record: dict[str, Any] = {
                    "id": store_id,
                    "name": body.name,
                    "backend": body.backend,
                    "scope": body.scope,
                    "metadata": body.metadata,
                    "created_at": now,
                }
                (stores_dir / f"{store_id}.json").write_text(json.dumps(store_record))

            except MemoryStoreInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "name": body.name,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = MemoryStoreCreateError(
                    message=f"Failed to create memory store: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "name": body.name,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from exc

        return JSONResponse(content=store_record, status_code=201)

    @router.post("/v1/memory_stores/{store_id}/query_runs")
    async def query_memory_store(store_id: str, body: MemoryStoreQueryRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "memory_store.query",
            attributes={
                "memory_store.id": store_id,
                "memory_store.query": body.query,
                "memory_store.scope": body.scope or "",
                "memory_store.bm25_weight": body.bm25_weight,
                "memory_store.vector_weight": body.vector_weight,
                "memory_store.rrf_k": body.rrf_k,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="memory_store.query.invocation",
                    code="memory_store_query",
                    timestamp=now,
                ),
            )

            try:
                store_path = stores_dir / f"{store_id}.json"
                if not store_path.exists():
                    raise MemoryStoreNotFoundError(
                        message=f"Memory store '{store_id}' not found",
                        timestamp=now,
                    )

                kb_store = KbStore(stores_dir / store_id / "chunks.db")
                bm25_results = kb_store.bm25_search(body.query, body.scope, body.limit)
                vector_results = kb_store.vector_search(body.query, body.scope, body.limit)
                results = _weighted_rrf_fuse(
                    [(bm25_results, body.bm25_weight), (vector_results, body.vector_weight)],
                    body.limit,
                    k=body.rrf_k,
                )
                span.set_attribute("memory_store.result_count", len(results))

            except MemoryStoreNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.query.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "query": body.query,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = MemoryStoreQueryError(
                    message=f"Memory store query failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.query.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "query": body.query,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from exc

        return JSONResponse(
            content={
                "results": results,
                "query": body.query,
                "scope": body.scope,
                "count": len(results),
                "store_id": store_id,
                "bm25_weight": body.bm25_weight,
                "vector_weight": body.vector_weight,
                "rrf_k": body.rrf_k,
            }
        )

    @router.post("/v1/memory_stores/{store_id}/write", status_code=201)
    async def write_memory(store_id: str, body: MemoryStoreWriteRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        embedder_id = body.embedder_id or "hash-128"
        scope = body.scope or "global"

        # Initialise to safe defaults; mutated inside the try block.
        action = "inserted"
        effective_content = body.content
        dialectic_label: _DialecticLabel | None = None
        dialectic_match_key: str | None = None

        with tracer.start_as_current_span(
            "memory_store.write",
            attributes={
                "memory_store.id": store_id,
                "memory_store.write.key": body.key,
                "memory_store.scope": scope,
                "memory_store.write.embedder_id": embedder_id,
                "memory_store.write.content_length": len(body.content),
                "memory_store.write.dialectic": body.dialectic,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="memory_store.write.invocation",
                    code="memory_store_write",
                    timestamp=now,
                ),
            )

            try:
                store_path = stores_dir / f"{store_id}.json"
                if not store_path.exists():
                    raise MemoryStoreNotFoundError(
                        message=f"Memory store '{store_id}' not found",
                        timestamp=now,
                    )

                kb_store = KbStore(stores_dir / store_id / "chunks.db")

                if body.dialectic and model_router is not None:
                    # 1. Retrieve top-K similar memories, excluding the incoming key.
                    bm25_r = kb_store.bm25_search(body.content, scope, body.dialectic_top_k)
                    vec_r = kb_store.vector_search(body.content, scope, body.dialectic_top_k)
                    candidates = _weighted_rrf_fuse(
                        [(bm25_r, 1.0), (vec_r, 1.0)],
                        body.dialectic_top_k,
                    )
                    candidates = [c for c in candidates if c["file_path"] != body.key]

                    # 2. Ask classifier to categorise the incoming content.
                    try:
                        result = await _classify_memory(
                            model_router, body.key, body.content, candidates
                        )
                    except Exception as exc:
                        raise MemoryStoreDialecticError(
                            message=f"Memory dialectic classification failed: {exc}",
                            timestamp=_now(),
                            cause=exc,
                        ) from exc

                    dialectic_label = result.label
                    dialectic_match_key = result.match_key
                    span.set_attribute("memory_store.write.dialectic_label", result.label)

                    # 3. Apply outcome.
                    if result.label == "duplicate":
                        # Skip write; content unchanged in store.
                        action = "deduplicated"
                        effective_content = body.content

                    elif result.label == "refinement":
                        effective_content = result.merged_content or body.content
                        kb_store.upsert_chunks(
                            body.key,
                            scope,
                            [
                                Chunk(
                                    file_path=body.key,
                                    kind="text",
                                    content=effective_content,
                                    start_line=0,
                                    end_line=0,
                                )
                            ],
                        )
                        action = "merged"

                    elif result.label == "contradiction":
                        effective_content = body.content
                        kb_store.upsert_chunks(
                            body.key,
                            scope,
                            [
                                Chunk(
                                    file_path=body.key,
                                    kind="text",
                                    content=effective_content,
                                    start_line=0,
                                    end_line=0,
                                )
                            ],
                        )
                        # Provenance edge: record what was superseded.
                        prov_dir = stores_dir / store_id / "provenance"
                        prov_dir.mkdir(parents=True, exist_ok=True)
                        (prov_dir / f"{body.key}.json").write_text(
                            json.dumps(
                                {
                                    "key": body.key,
                                    "superseded_at": now,
                                    "superseded_match_key": result.match_key,
                                    "explanation": result.explanation,
                                }
                            )
                        )
                        action = "superseded"

                    else:  # net-new
                        effective_content = body.content
                        was_present = kb_store.has_key(body.key)
                        kb_store.upsert_chunks(
                            body.key,
                            scope,
                            [
                                Chunk(
                                    file_path=body.key,
                                    kind="text",
                                    content=effective_content,
                                    start_line=0,
                                    end_line=0,
                                )
                            ],
                        )
                        action = "updated" if was_present else "inserted"

                else:
                    # No dialectic: existing upsert behaviour.
                    effective_content = body.content
                    was_present = kb_store.has_key(body.key)
                    kb_store.upsert_chunks(
                        body.key,
                        scope,
                        [
                            Chunk(
                                file_path=body.key,
                                kind="text",
                                content=effective_content,
                                start_line=0,
                                end_line=0,
                            )
                        ],
                    )
                    action = "updated" if was_present else "inserted"

                span.set_attribute("memory_store.write.action", action)

                if event_log is not None:
                    event_data: dict[str, Any] = {
                        "store_id": store_id,
                        "key": body.key,
                        "scope": scope,
                        "action": action,
                    }
                    if action == "superseded" and dialectic_match_key is not None:
                        event_data["superseded_memory_id"] = dialectic_match_key
                    try:
                        await event_log.append(store_id, "memory.write", event_data)
                    except Exception as exc:
                        err3 = MemoryStoreWriteError(
                            message=f"Failed to record memory.write event: {exc}",
                            timestamp=_now(),
                            cause=exc,
                        )
                        record_error(span, err3)
                        audit_log.write(
                            AuditLogEntry(
                                level="error",
                                event="memory_store.write.event_log_failed",
                                code=err3.code,
                                timestamp=err3.timestamp,
                                detail={
                                    "memory_store_id": store_id,
                                    "key": body.key,
                                    "action": action,
                                    "message": err3.message,
                                },
                            )
                        )
                        raise err3 from exc

            except MemoryStoreNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.write.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "key": body.key,
                            "message": err.message,
                        },
                    )
                )
                raise

            except MemoryStoreDialecticError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.write.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "key": body.key,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = MemoryStoreWriteError(
                    message=f"Memory store write failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="memory_store.write.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "memory_store_id": store_id,
                            "key": body.key,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from exc

        return JSONResponse(
            content={
                "store_id": store_id,
                "key": body.key,
                "content": effective_content,
                "scope": scope,
                "embedder_id": embedder_id,
                "action": action,
                "created_at": now,
                "dialectic_label": dialectic_label,
                "dialectic_match_key": dialectic_match_key,
            },
            status_code=201,
        )

    return router
