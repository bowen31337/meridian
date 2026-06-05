import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { SessionEvent } from "../api/client.js";
import { useMeridianApi } from "../api/context.js";
import { getTracer, recordInvocationEvent, recordStreamError } from "./telemetry.js";

// ---------------------------------------------------------------------------
// Payload shapes (inferred from event kind; payload is open-ended in the SDK)
// ---------------------------------------------------------------------------

interface TextBlock {
  type: "text";
  text: string;
}

interface ThinkingBlock {
  type: "thinking";
  thinking: string;
}

interface ToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
}

type ContentBlock =
  | TextBlock
  | ThinkingBlock
  | ToolUseBlock
  | { type: string; [key: string]: unknown };

interface MessagePayload {
  role: "user" | "assistant" | "system";
  content: string | ContentBlock[];
  sequence?: number;
}

interface ToolCallPayload {
  tool_call_id: string;
  name: string;
  input: Record<string, unknown>;
}

interface ToolResultPayload {
  tool_call_id: string;
  content: unknown;
  is_error?: boolean;
}

interface ErrorPayload {
  message: string;
  code?: string;
}

// ---------------------------------------------------------------------------
// Phase
// ---------------------------------------------------------------------------

type SessionPhase =
  | "idle"
  | "waiting_for_model"
  | "waiting_for_tool"
  | "waiting_for_user"
  | "error";

const PHASE_LABELS: Record<SessionPhase, string> = {
  idle: "Idle",
  waiting_for_model: "Waiting for model…",
  waiting_for_tool: "Waiting for tool…",
  waiting_for_user: "Waiting for user",
  error: "Error",
};

function derivePhase(events: SessionEvent[]): SessionPhase {
  const last = events[events.length - 1];
  if (last === undefined) return "idle";
  switch (last.kind) {
    case "message": {
      const p = last.payload as Partial<MessagePayload>;
      if (p.role === "user") return "waiting_for_model";
      if (p.role === "assistant") return "waiting_for_user";
      return "idle";
    }
    case "tool_call":
      return "waiting_for_tool";
    case "tool_result":
      return "waiting_for_model";
    case "error":
      return "error";
    default:
      return "idle";
  }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function PhaseIndicator({ phase }: { phase: SessionPhase }) {
  return (
    <div data-testid="phase-indicator" data-phase={phase}>
      {PHASE_LABELS[phase]}
    </div>
  );
}

function ThinkingBlockView({ thinking }: { thinking: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <details
      data-testid="thinking-block"
      open={expanded}
      onToggle={(e) => setExpanded((e.target as HTMLDetailsElement).open)}
    >
      <summary data-testid="thinking-block-summary">Thinking…</summary>
      <pre data-testid="thinking-block-content">{thinking}</pre>
    </details>
  );
}

function ContentBlocksView({ blocks }: { blocks: ContentBlock[] }) {
  return (
    <>
      {blocks.map((block, i) => {
        if (block.type === "thinking") {
          // biome-ignore lint/suspicious/noArrayIndexKey: content blocks are a static, append-only list and are never reordered
          return <ThinkingBlockView key={i} thinking={(block as ThinkingBlock).thinking} />;
        }
        if (block.type === "text") {
          return (
            // biome-ignore lint/suspicious/noArrayIndexKey: content blocks are a static, append-only list and are never reordered
            <p key={i} data-testid={`text-block-${i}`}>
              {(block as TextBlock).text}
            </p>
          );
        }
        if (block.type === "tool_use") {
          const tb = block as ToolUseBlock;
          return (
            <div key={tb.id} data-testid={`inline-tool-use-${tb.id}`}>
              <em>Tool use: {tb.name}</em>
            </div>
          );
        }
        return (
          // biome-ignore lint/suspicious/noArrayIndexKey: content blocks are a static, append-only list and are never reordered
          <div key={i} data-testid={`unknown-block-${i}`}>
            <pre>{JSON.stringify(block, null, 2)}</pre>
          </div>
        );
      })}
    </>
  );
}

function MessageBlock({ event }: { event: SessionEvent }) {
  const payload = event.payload as Partial<MessagePayload>;
  const role = payload.role ?? "unknown";
  const content = payload.content;

  return (
    <div data-testid={`message-${event.id}`} data-kind="message" data-role={role}>
      <span data-testid="message-role">{role}</span>
      <div data-testid="message-content">
        {typeof content === "string" || content == null ? (
          <p>{content ?? ""}</p>
        ) : (
          <ContentBlocksView blocks={content} />
        )}
      </div>
    </div>
  );
}

function ToolCallBlock({
  event,
  resultEvent,
}: {
  event: SessionEvent;
  resultEvent: SessionEvent | undefined;
}) {
  const payload = event.payload as Partial<ToolCallPayload>;
  const resultPayload = resultEvent?.payload as Partial<ToolResultPayload> | undefined;

  return (
    <div data-testid={`tool-call-${payload.tool_call_id ?? event.id}`} data-kind="tool_call">
      <strong data-testid="tool-call-name">{payload.name ?? "(unknown)"}</strong>
      <pre data-testid="tool-call-input">{JSON.stringify(payload.input ?? {}, null, 2)}</pre>
      {resultPayload !== undefined && (
        <div
          data-testid={`tool-result-${payload.tool_call_id ?? event.id}`}
          data-kind="tool_result"
        >
          {resultPayload.is_error && <span data-testid="tool-result-error-badge">Error</span>}
          <pre data-testid="tool-result-content">
            {typeof resultPayload.content === "string"
              ? resultPayload.content
              : JSON.stringify(resultPayload.content ?? null, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SessionViewer
// ---------------------------------------------------------------------------

export interface SessionViewerProps {
  readonly sessionId: string;
}

export function SessionViewer({ sessionId }: SessionViewerProps): React.ReactElement {
  const { baseUrl, auditLog } = useMeridianApi();
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(true);

  const lastProcessedCountRef = useRef(0);

  useEffect(() => {
    const url = `${baseUrl}/v1/sessions/${encodeURIComponent(sessionId)}/events?stream=true`;
    const es = new EventSource(url);
    setIsConnecting(true);
    setStreamError(null);

    es.onopen = () => {
      setIsConnecting(false);
    };

    es.onmessage = (e: MessageEvent<string>) => {
      setIsConnecting(false);
      try {
        const event = JSON.parse(e.data) as SessionEvent;
        setEvents((prev) => [...prev, event]);
      } catch {
        // ignore malformed frames
      }
    };

    es.onerror = () => {
      const msg = "Session stream disconnected";
      setStreamError(msg);
      setIsConnecting(false);
      const err = new Error(msg);
      const tracer = getTracer();
      tracer.startActiveSpan("session_viewer.stream", (span) => {
        recordStreamError(span, err, auditLog, { sessionId });
        span.end();
      });
      es.close();
    };

    return () => {
      es.close();
    };
  }, [sessionId, baseUrl, auditLog]);

  // Emit OTel invocation span each time the event count grows.
  useEffect(() => {
    if (events.length === lastProcessedCountRef.current) return;
    const tracer = getTracer();
    const timestamp = new Date().toISOString();
    tracer.startActiveSpan(
      "session_viewer.process",
      { attributes: { "session.id": sessionId, "event.count": events.length } },
      (span) => {
        recordInvocationEvent(span, { sessionId, timestamp, eventCount: events.length });
        span.end();
      },
    );
    lastProcessedCountRef.current = events.length;
  }, [events.length, sessionId]);

  const phase = derivePhase(events);

  // Pre-compute which tool_result events are inlined under their tool_call.
  const { toolResultsByCallId, handledResultEventIds } = useMemo(() => {
    const byCallId = new Map<string, SessionEvent>();
    for (const event of events) {
      if (event.kind === "tool_result") {
        const p = event.payload as Partial<ToolResultPayload>;
        if (p.tool_call_id) byCallId.set(p.tool_call_id, event);
      }
    }
    const handledIds = new Set(
      events
        .filter((e) => e.kind === "tool_call")
        .flatMap((e) => {
          const p = e.payload as Partial<ToolCallPayload>;
          if (!p.tool_call_id) return [];
          const result = byCallId.get(p.tool_call_id);
          return result ? [result.id] : [];
        }),
    );
    return { toolResultsByCallId: byCallId, handledResultEventIds: handledIds };
  }, [events]);

  if (isConnecting && events.length === 0) {
    return <div data-testid="session-viewer-connecting">Connecting…</div>;
  }

  if (streamError && events.length === 0) {
    return (
      <div role="alert" data-testid="session-viewer-error">
        {streamError}
      </div>
    );
  }

  return (
    <div data-testid="session-viewer">
      <PhaseIndicator phase={phase} />
      {streamError && (
        <div role="alert" data-testid="session-viewer-error">
          {streamError}
        </div>
      )}
      <div data-testid="event-timeline">
        {events.map((event) => {
          if (event.kind === "message") {
            return <MessageBlock key={event.id} event={event} />;
          }

          if (event.kind === "tool_call") {
            const p = event.payload as Partial<ToolCallPayload>;
            const resultEvent = p.tool_call_id
              ? toolResultsByCallId.get(p.tool_call_id)
              : undefined;
            return <ToolCallBlock key={event.id} event={event} resultEvent={resultEvent} />;
          }

          if (event.kind === "tool_result") {
            if (handledResultEventIds.has(event.id)) return null;
            const p = event.payload as Partial<ToolResultPayload>;
            return (
              <div
                key={event.id}
                data-testid={`orphan-tool-result-${event.id}`}
                data-kind="tool_result"
              >
                <pre>
                  {typeof p.content === "string"
                    ? p.content
                    : JSON.stringify(p.content ?? null, null, 2)}
                </pre>
              </div>
            );
          }

          if (event.kind === "error") {
            const p = event.payload as Partial<ErrorPayload>;
            return (
              <div
                key={event.id}
                data-testid={`error-event-${event.id}`}
                data-kind="error"
                role="alert"
              >
                {p.message ?? "Unknown error"}
              </div>
            );
          }

          return null;
        })}
      </div>
    </div>
  );
}
