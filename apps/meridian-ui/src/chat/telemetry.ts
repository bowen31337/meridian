import { SpanStatusCode, trace } from "@opentelemetry/api";
import type { Span } from "@opentelemetry/api";
import type { AuditLog } from "../workspace/audit.js";
import type { AuditLogEntry } from "../workspace/types.js";

const TRACER_NAME = "meridian.chat_composer";
const TRACER_VERSION = "0.1.0";

export function getTracer() {
  return trace.getTracer(TRACER_NAME, TRACER_VERSION);
}

export interface ComposerInvocationEvent {
  readonly sessionId: string;
  readonly timestamp: string;
  readonly channelId?: string;
  readonly agentId?: string;
  readonly contentLength: number;
}

export function recordComposerInvocationEvent(span: Span, event: ComposerInvocationEvent): void {
  span.addEvent("chat_composer.invocation", {
    "session.id": event.sessionId,
    "event.timestamp": event.timestamp,
    "message.content_length": event.contentLength,
    ...(event.channelId ? { "channel.id": event.channelId } : {}),
    ...(event.agentId ? { "agent.id": event.agentId } : {}),
  });
}

export function recordComposerFailure(
  span: Span,
  err: unknown,
  auditLog: AuditLog,
  context: { sessionId: string; operation: string },
): void {
  const message = err instanceof Error ? err.message : String(err);
  span.setStatus({ code: SpanStatusCode.ERROR, message });
  if (err instanceof Error) span.recordException(err);

  const entry: AuditLogEntry = {
    level: "error",
    event: `chat_composer.${context.operation}.failed`,
    sessionId: context.sessionId,
    timestamp: new Date().toISOString(),
    detail: { message },
  };
  auditLog.write(entry);
}
