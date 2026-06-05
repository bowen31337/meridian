"""
Cursor pagination conformance suite.

Tests cover:
  - encode_cursor / decode_cursor round-trip preserves (created_at, id).
  - decode_cursor raises CursorDecodeError on invalid base64 input.
  - decode_cursor raises CursorDecodeError on valid base64 but bad payload.
  - apply_cursor_filter returns items after the cursor position.
  - apply_cursor_filter returns empty list when cursor is the last item.
  - apply_cursor_filter falls back to tuple comparison when cursor item deleted.
  - make_cursor_page returns full page and None next_cursor when items <= limit.
  - make_cursor_page returns sliced page and non-None next_cursor when items > limit.
  - make_cursor_page next_cursor encodes the last item on the page.
  - build_link_header returns RFC 8288 rel=next value with cursor and limit.
  - Middleware: limit > 200 returns 422 with code cursor_limit_exceeded.
  - Middleware: limit < 1 returns 422 with code cursor_limit_exceeded.
  - Middleware: non-integer limit returns 422 with code cursor_limit_exceeded.
  - Middleware: limit rejection writes audit log entry.
  - Middleware: valid request passes through unchanged when no X-Next-Cursor.
  - Middleware: X-Next-Cursor is converted to RFC 8288 Link header.
  - Middleware: X-Next-Cursor is stripped from final response.
  - GET /v1/skills default limit is 50.
  - GET /v1/skills next_cursor is null when all items fit on one page.
  - GET /v1/skills next_cursor is present when more pages exist.
  - GET /v1/skills Link header present when next page exists.
  - GET /v1/skills Link header absent when no next page.
  - GET /v1/skills cursor param navigates to next page.
  - GET /v1/skills invalid cursor returns 400 with code cursor_invalid.
  - GET /v1/skills limit > 200 returns 422 with code cursor_limit_exceeded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._pagination import (
    CursorDecodeError,
    apply_cursor_filter,
    build_link_header,
    decode_cursor,
    encode_cursor,
    make_cursor_page,
)
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _create_skill(
    client: TestClient,
    *,
    name: str = "test-skill",
) -> dict[str, Any]:
    resp = client.post(
        "/v1/skills",
        json={
            "name": name,
            "description": "A test skill",
            "instructions": "Do the thing",
            "tools": [{"name": "bash", "description": "Run shell commands"}],
        },
    )
    assert resp.status_code == 201
    return resp.json()


def _read_audit_log(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Unit: encode_cursor / decode_cursor
# ---------------------------------------------------------------------------


class TestEncodeDecode:
    def test_round_trip(self) -> None:
        cursor = encode_cursor("2024-01-01T00:00:00+00:00", "skill_abc123")
        created_at, record_id = decode_cursor(cursor, timestamp="t")
        assert created_at == "2024-01-01T00:00:00+00:00"
        assert record_id == "skill_abc123"

    def test_opaque_bytes(self) -> None:
        cursor = encode_cursor("2024-01-01T00:00:00+00:00", "skill_abc123")
        assert "2024" not in cursor
        assert "skill_abc123" not in cursor

    def test_invalid_base64_raises(self) -> None:
        with pytest.raises(CursorDecodeError) as exc_info:
            decode_cursor("not!!valid!!base64", timestamp="t")
        assert exc_info.value.code == "cursor_invalid"

    def test_valid_base64_bad_payload_raises(self) -> None:
        import base64

        bad = base64.urlsafe_b64encode(b'{"x":1}').decode()
        with pytest.raises(CursorDecodeError):
            decode_cursor(bad, timestamp="t")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(CursorDecodeError):
            decode_cursor("", timestamp="t")


# ---------------------------------------------------------------------------
# Unit: apply_cursor_filter
# ---------------------------------------------------------------------------


class TestApplyCursorFilter:
    def _items(self) -> list[dict[str, Any]]:
        return [
            {"created_at": "2024-03-01", "id": "c"},
            {"created_at": "2024-02-01", "id": "b"},
            {"created_at": "2024-01-01", "id": "a"},
        ]

    def test_returns_items_after_cursor(self) -> None:
        items = self._items()
        result = apply_cursor_filter(items, "2024-03-01", "c")
        assert [i["id"] for i in result] == ["b", "a"]

    def test_cursor_on_last_item_returns_empty(self) -> None:
        items = self._items()
        result = apply_cursor_filter(items, "2024-01-01", "a")
        assert result == []

    def test_cursor_on_middle_item(self) -> None:
        items = self._items()
        result = apply_cursor_filter(items, "2024-02-01", "b")
        assert [i["id"] for i in result] == ["a"]

    def test_deleted_cursor_item_falls_back(self) -> None:
        items = self._items()
        # Cursor points to a deleted item between b and a
        result = apply_cursor_filter(items, "2024-01-15", "deleted")
        # Items with (created_at, id) < ("2024-01-15", "deleted") — only "a"
        assert [i["id"] for i in result] == ["a"]


# ---------------------------------------------------------------------------
# Unit: make_cursor_page
# ---------------------------------------------------------------------------


class TestMakeCursorPage:
    def _item(self, ts: str, id_: str) -> dict[str, Any]:
        return {"created_at": ts, "id": id_}

    def test_all_items_fit_returns_null_next_cursor(self) -> None:
        items = [self._item("2024-01-01", "a"), self._item("2024-01-02", "b")]
        page, next_cursor = make_cursor_page(items, limit=5)
        assert len(page) == 2
        assert next_cursor is None

    def test_excess_items_returns_non_null_next_cursor(self) -> None:
        items = [self._item("2024-01-0" + str(i), f"id{i}") for i in range(1, 4)]
        page, next_cursor = make_cursor_page(items, limit=2)
        assert len(page) == 2
        assert next_cursor is not None

    def test_next_cursor_encodes_last_page_item(self) -> None:
        items = [
            self._item("2024-03-01", "c"),
            self._item("2024-02-01", "b"),
            self._item("2024-01-01", "a"),
        ]
        page, next_cursor = make_cursor_page(items, limit=2)
        assert next_cursor is not None
        created_at, record_id = decode_cursor(next_cursor, timestamp="t")
        assert created_at == page[-1]["created_at"]
        assert record_id == page[-1]["id"]

    def test_empty_items_returns_null_next_cursor(self) -> None:
        page, next_cursor = make_cursor_page([], limit=10)
        assert page == []
        assert next_cursor is None

    def test_exact_limit_returns_null_next_cursor(self) -> None:
        items = [self._item("2024-01-01", "a"), self._item("2024-01-02", "b")]
        page, next_cursor = make_cursor_page(items, limit=2)
        assert next_cursor is None


# ---------------------------------------------------------------------------
# Unit: build_link_header
# ---------------------------------------------------------------------------


class TestBuildLinkHeader:
    def test_rel_next_format(self) -> None:
        link = build_link_header("http://testserver/v1/skills", "abc123", 50)
        assert 'rel="next"' in link

    def test_contains_cursor(self) -> None:
        link = build_link_header("http://testserver/v1/skills", "abc123", 50)
        assert "cursor=abc123" in link

    def test_contains_limit(self) -> None:
        link = build_link_header("http://testserver/v1/skills", "abc123", 10)
        assert "limit=10" in link

    def test_angle_bracket_url(self) -> None:
        link = build_link_header("http://testserver/v1/skills", "abc123", 50)
        assert link.startswith("<http://testserver/v1/skills?")
        assert link.endswith('>; rel="next"')


# ---------------------------------------------------------------------------
# Integration: middleware limit validation
# ---------------------------------------------------------------------------


class TestMiddlewareLimitValidation:
    def test_limit_above_max_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/skills", params={"limit": 201})
        assert resp.status_code == 422

    def test_limit_at_max_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/skills", params={"limit": 200})
        assert resp.status_code == 200

    def test_limit_zero_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/skills", params={"limit": 0})
        assert resp.status_code == 422

    def test_limit_negative_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/skills", params={"limit": -1})
        assert resp.status_code == 422

    def test_non_integer_limit_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/skills?limit=abc")
        assert resp.status_code == 422

    def test_limit_rejection_has_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills", params={"limit": 999}).json()
        assert body["error"]["code"] == "cursor_limit_exceeded"

    def test_limit_rejection_writes_audit_log(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/skills", params={"limit": 999})
        entries = _read_audit_log(storage_root)
        codes = [e["code"] for e in entries]
        assert "cursor_limit_exceeded" in codes


# ---------------------------------------------------------------------------
# Integration: middleware Link header injection
# ---------------------------------------------------------------------------


class TestMiddlewareLinkHeader:
    def test_link_header_present_when_next_page(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, name="skill-a")
        _create_skill(client, name="skill-b")
        resp = client.get("/v1/skills", params={"limit": 1})
        assert "link" in {k.lower() for k in resp.headers}

    def test_link_header_contains_rel_next(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, name="skill-a")
        _create_skill(client, name="skill-b")
        resp = client.get("/v1/skills", params={"limit": 1})
        link = resp.headers.get("link", "")
        assert 'rel="next"' in link

    def test_link_header_absent_on_last_page(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client)
        resp = client.get("/v1/skills", params={"limit": 50})
        assert "link" not in {k.lower() for k in resp.headers}

    def test_x_next_cursor_not_in_final_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, name="skill-a")
        _create_skill(client, name="skill-b")
        resp = client.get("/v1/skills", params={"limit": 1})
        assert "x-next-cursor" not in {k.lower() for k in resp.headers}


# ---------------------------------------------------------------------------
# Integration: GET /v1/skills cursor pagination
# ---------------------------------------------------------------------------


class TestSkillsEndpointCursorPagination:
    def test_default_limit_is_50(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills").json()
        assert body["limit"] == 50

    def test_next_cursor_null_when_all_fit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client)
        body = client.get("/v1/skills", params={"limit": 10}).json()
        assert body["next_cursor"] is None

    def test_next_cursor_present_when_more(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _create_skill(client, name="skill-a")
        _create_skill(client, name="skill-b")
        body = client.get("/v1/skills", params={"limit": 1}).json()
        assert body["next_cursor"] is not None

    def test_cursor_traverses_all_items(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        for i in range(5):
            _create_skill(client, name=f"skill-{i}")

        seen_ids: list[str] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 2}
            if cursor:
                params["cursor"] = cursor
            body = client.get("/v1/skills", params=params).json()
            seen_ids.extend(item["id"] for item in body["items"])
            cursor = body["next_cursor"]
            if cursor is None:
                break

        assert len(seen_ids) == 5
        assert len(set(seen_ids)) == 5

    def test_invalid_cursor_returns_400(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/skills", params={"cursor": "not-valid!!"})
        assert resp.status_code == 400

    def test_invalid_cursor_returns_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/skills", params={"cursor": "not-valid!!"}).json()
        assert body["error"]["code"] == "cursor_invalid"

    def test_limit_above_200_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/skills", params={"limit": 201})
        assert resp.status_code == 422
