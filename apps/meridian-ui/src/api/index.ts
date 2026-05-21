export type {
  ApiClient,
  Channel,
  ChannelList,
  CreateSessionRequest,
  ErrorBody,
  ListSessionEventsParams,
  ListSessionsParams,
  Message,
  Provider,
  ProviderList,
  SendMessageRequest,
  Session,
  SessionCreateBody,
  SessionEvent,
  SessionEventKind,
  SessionEventList,
  SessionList,
} from "./client.js";
export { ApiError, createApiClient } from "./client.js";

export type { MeridianApiContextValue, MeridianApiProviderProps } from "./context.js";
export { MeridianApiProvider, useMeridianApi } from "./context.js";

export { queryKeys } from "./query-keys.js";

export type { ApiInvocationEvent } from "./telemetry.js";

export { useListSessions, useGetSession, useCreateSession, useCloseSession } from "./hooks/useSessions.js";
export { useListProviders } from "./hooks/useProviders.js";
export { useListSessionEvents } from "./hooks/useSessionEvents.js";
export { useSendMessage } from "./hooks/useMessages.js";
export type { SendMessageArgs } from "./hooks/useMessages.js";
export { useListChannels } from "./hooks/useChannels.js";
