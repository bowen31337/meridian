from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# ─── Content block types ──────────────────────────────────────────────────────


class CacheControl(BaseModel):
    type: Literal["ephemeral"] = "ephemeral"


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    cache_control: CacheControl | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[Any]


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str


ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock,
    Field(discriminator="type"),
]

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    role: MessageRole
    content: str | list[ContentBlock]


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


# ─── Call options ─────────────────────────────────────────────────────────────


class ModelCallOpts(BaseModel):
    model: str
    messages: list[Message]
    system: str | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    max_tokens: int = 8096
    temperature: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    role: str | None = None
    skill_id: str | None = None
    estimated_input_tokens: int | None = None
    session_id: str | None = None
    # Capability-gated options — stripped by Router if provider lacks the capability
    stream: bool = True
    enable_thinking: bool = False
    thinking_budget_tokens: int | None = None


# ─── Streaming event types ────────────────────────────────────────────────────


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    model: str
    provider: str
    input_tokens: int | None = None


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDeltaEvent(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class ToolUseStartEvent(BaseModel):
    type: Literal["tool_use_start"] = "tool_use_start"
    id: str
    name: str


class ToolInputDeltaEvent(BaseModel):
    type: Literal["tool_input_delta"] = "tool_input_delta"
    id: str
    partial_json: str


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    stop_reason: str | None = None


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


ModelEvent = Annotated[
    MessageStartEvent
    | TextDeltaEvent
    | ThinkingDeltaEvent
    | ToolUseStartEvent
    | ToolInputDeltaEvent
    | MessageDeltaEvent
    | MessageStopEvent,
    Field(discriminator="type"),
]


# ─── Token counting ───────────────────────────────────────────────────────────


class ModelCountReq(BaseModel):
    model: str
    messages: list[Message]
    system: str | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)


class TokenCount(BaseModel):
    input_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
