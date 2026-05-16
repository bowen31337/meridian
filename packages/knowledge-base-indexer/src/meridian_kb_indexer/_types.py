from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ChunkKind = Literal["symbol", "heading", "text"]
IndexEventKind = Literal["created", "modified", "deleted", "initial_scan"]


class Chunk(BaseModel):
    """A parsed unit of content from a workspace file."""

    file_path: str
    kind: ChunkKind
    content: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    symbol_kind: str | None = None  # "function", "class", "method", "interface", "type"
    heading_level: int | None = None  # 1–6 for ATX headings
    heading_text: str | None = None
    language: str | None = None


class IndexEvent(BaseModel):
    """Fired when a file in the workspace is created, modified, or deleted."""

    event_kind: IndexEventKind
    file_path: str
    chunks: list[Chunk] = Field(default_factory=list)
    error: str | None = None
    timestamp: str


class IndexerError(Exception):
    """Raised by WorkspaceIndexer when a file cannot be read or parsed."""

    def __init__(self, code: str, message: str, file_path: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.file_path = file_path
