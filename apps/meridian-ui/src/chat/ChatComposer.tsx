import React, { useCallback, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Message, Session } from "../api/client.js";
import { createApiClient } from "../api/client.js";
import { useMeridianApi } from "../api/context.js";
import { useListChannels } from "../api/hooks/useChannels.js";
import { useSendMessage } from "../api/hooks/useMessages.js";
import { useCreateSession } from "../api/hooks/useSessions.js";
import { queryKeys } from "../api/query-keys.js";
import { getTracer, recordComposerFailure, recordComposerInvocationEvent } from "./telemetry.js";

export interface ChatComposerProps {
  /** If provided, messages are sent to this existing session. */
  sessionId?: string;
  onMessageSent?: (message: Message) => void;
  /** Called when a new session is created (only relevant when sessionId is not provided). */
  onSessionCreated?: (session: Session) => void;
}

export function ChatComposer({ sessionId, onMessageSent, onSessionCreated }: ChatComposerProps) {
  const { auditLog, baseUrl } = useMeridianApi();

  const [content, setContent] = useState("");
  const [selectedAgentId, setSelectedAgentId] = useState("");
  const [selectedChannelId, setSelectedChannelId] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const { data: channelsData } = useListChannels();
  const channels = channelsData?.items ?? [];

  const { data: agentsData } = useQuery({
    queryKey: queryKeys.agents.list(),
    queryFn: () => createApiClient(baseUrl).listAgents(),
    enabled: !sessionId,
  });
  const agents = agentsData?.items ?? [];

  const createSessionMutation = useCreateSession();
  const sendMessageMutation = useSendMessage();

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = content.trim();
      if (!trimmed) return;

      setErrorMessage(null);

      const timestamp = new Date().toISOString();
      const tracer = getTracer();

      // Track resolved session ID in outer scope so the catch block can log it.
      let resolvedSessionId = sessionId ?? "";

      await tracer.startActiveSpan("chat_composer.submit", async (span) => {
        try {
          if (!resolvedSessionId) {
            const newSession = await createSessionMutation.mutateAsync({
              provider: "anthropic",
              model: "claude-3-5-sonnet",
              ...(selectedAgentId ? { agent_id: selectedAgentId } : {}),
            });
            resolvedSessionId = newSession.id;
            onSessionCreated?.(newSession);
          }

          recordComposerInvocationEvent(span, {
            sessionId: resolvedSessionId,
            timestamp,
            contentLength: trimmed.length,
            ...(selectedChannelId ? { channelId: selectedChannelId } : {}),
            ...(selectedAgentId ? { agentId: selectedAgentId } : {}),
          });

          const message = await sendMessageMutation.mutateAsync({
            sessionId: resolvedSessionId,
            body: {
              content: trimmed,
              ...(selectedChannelId ? { channel_id: selectedChannelId } : {}),
            },
          });

          span.end();
          setContent("");
          onMessageSent?.(message);
        } catch (err) {
          recordComposerFailure(span, err, auditLog, {
            sessionId: resolvedSessionId,
            operation: "submit",
          });
          span.end();
          const msg = err instanceof Error ? err.message : "Failed to send message";
          setErrorMessage(msg);
        }
      });
    },
    [
      content,
      sessionId,
      selectedAgentId,
      selectedChannelId,
      createSessionMutation,
      sendMessageMutation,
      onMessageSent,
      onSessionCreated,
      auditLog,
    ],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void handleSubmit(e as unknown as React.FormEvent);
      }
    },
    [handleSubmit],
  );

  const isSubmitting = createSessionMutation.isPending || sendMessageMutation.isPending;
  // When no session exists, require an agent to be selected (if any agents are available).
  const agentRequired = !sessionId && agents.length > 0 && !selectedAgentId;
  const canSubmit = content.trim().length > 0 && !isSubmitting && !agentRequired;

  return (
    <form data-testid="chat-composer" onSubmit={handleSubmit}>
      {!sessionId && (
        <div data-testid="agent-picker">
          <label htmlFor="agent-select">Agent</label>
          <select
            id="agent-select"
            data-testid="agent-select"
            value={selectedAgentId}
            onChange={(e) => setSelectedAgentId(e.target.value)}
          >
            <option value="">Select an agent…</option>
            {agents.map((agent) => (
              <option key={agent.id} value={agent.id}>
                {agent.name}
              </option>
            ))}
          </select>
        </div>
      )}

      {channels.length > 0 && (
        <div data-testid="channel-picker">
          <label htmlFor="channel-select">Channel</label>
          <select
            id="channel-select"
            data-testid="channel-select"
            value={selectedChannelId}
            onChange={(e) => setSelectedChannelId(e.target.value)}
          >
            <option value="">Default channel</option>
            {channels.map((ch) => (
              <option key={ch.id} value={ch.id}>
                {ch.name}
              </option>
            ))}
          </select>
        </div>
      )}

      <textarea
        data-testid="message-input"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
        rows={3}
        disabled={isSubmitting}
      />

      {errorMessage && (
        <div role="alert" data-testid="composer-error">
          {errorMessage}
        </div>
      )}

      <button type="submit" data-testid="send-button" disabled={!canSubmit}>
        {isSubmitting ? "Sending…" : "Send"}
      </button>
    </form>
  );
}
