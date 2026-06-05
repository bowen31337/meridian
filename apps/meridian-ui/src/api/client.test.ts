import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, createApiClient } from "./client.js";

const BASE_URL = "http://api.test";
const client = createApiClient(BASE_URL);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ApiError", () => {
  it("uses body message when present", () => {
    const err = new ApiError(404, { code: "NOT_FOUND", message: "not found" });
    expect(err.message).toBe("not found");
    expect(err.status).toBe(404);
    expect(err.name).toBe("ApiError");
  });

  it("falls back to HTTP status when body is absent", () => {
    const err = new ApiError(500, undefined);
    expect(err.message).toBe("HTTP 500");
  });
});

describe("createApiClient", () => {
  it("listSessions fetches GET /sessions", async () => {
    const mockData = { sessions: [], total: 0 };
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(mockData));

    const result = await client.listSessions();
    expect(fetch).toHaveBeenCalledWith(`${BASE_URL}/sessions`, expect.objectContaining({}));
    expect(result).toEqual(mockData);
  });

  it("listSessions appends query params", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ sessions: [], total: 0 }));
    await client.listSessions({ limit: 10, offset: 20 });
    const url = vi.mocked(fetch).mock.calls[0]?.[0] as string;
    expect(url).toContain("limit=10");
    expect(url).toContain("offset=20");
  });

  it("createSession sends POST with JSON body", async () => {
    const session = {
      id: "s1",
      status: "active",
      provider: "anthropic",
      model: "claude-3",
      created_at: "",
      updated_at: "",
    } as const;
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(session, 201));

    const result = await client.createSession({ provider: "anthropic", model: "claude-3" });
    expect(fetch).toHaveBeenCalledWith(
      `${BASE_URL}/sessions`,
      expect.objectContaining({ method: "POST" }),
    );
    expect(result).toEqual(session);
  });

  it("getSession fetches GET /sessions/:id", async () => {
    const session = {
      id: "s1",
      status: "active",
      provider: "anthropic",
      model: "claude-3",
      created_at: "",
      updated_at: "",
    } as const;
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(session));

    const result = await client.getSession("s1");
    expect(fetch).toHaveBeenCalledWith(`${BASE_URL}/sessions/s1`, expect.objectContaining({}));
    expect(result).toEqual(session);
  });

  it("closeSession sends DELETE and returns undefined for 204", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(new Response(null, { status: 204 }));

    const result = await client.closeSession("s1");
    expect(fetch).toHaveBeenCalledWith(
      `${BASE_URL}/sessions/s1`,
      expect.objectContaining({ method: "DELETE" }),
    );
    expect(result).toBeUndefined();
  });

  it("listProviders fetches GET /providers", async () => {
    const data = { providers: [] };
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(data));

    const result = await client.listProviders();
    expect(fetch).toHaveBeenCalledWith(`${BASE_URL}/providers`, expect.objectContaining({}));
    expect(result).toEqual(data);
  });

  it("listSessionEvents fetches GET /sessions/:id/events", async () => {
    const data = { events: [], total: 0 };
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(data));

    const result = await client.listSessionEvents("s1");
    expect(fetch).toHaveBeenCalledWith(
      `${BASE_URL}/sessions/s1/events`,
      expect.objectContaining({}),
    );
    expect(result).toEqual(data);
  });

  it("throws ApiError for non-OK response", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "SERVER_ERROR", message: "oops" }, 500),
    );
    try {
      await client.listSessions();
      expect.fail("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      const err = e as ApiError;
      expect(err.status).toBe(500);
      expect(err.message).toBe("oops");
    }
  });

  it("throws ApiError with status text for non-JSON error body", async () => {
    vi.mocked(fetch).mockResolvedValue(new Response("Internal Server Error", { status: 500 }));
    try {
      await client.listSessions();
      expect.fail("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      const err = e as ApiError;
      expect(err.status).toBe(500);
      expect(err.body).toBeUndefined();
    }
  });
});
