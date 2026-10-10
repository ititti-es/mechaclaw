// Memory Core tests cover the conversation recall client.
import type { OpenClawConfig } from "openclaw/plugin-sdk/memory-core-host-runtime-core";
import { requestBodyText } from "openclaw/plugin-sdk/test-env";
import { describe, expect, it, vi } from "vitest";
import {
  parseConversationPath,
  rankByStoreRerank,
  readStoredConversation,
  resolveConversationRecallConfig,
  resolveConversationRecallForSession,
  searchStoredConversations,
  type ConversationRecallCall,
} from "./conversation-recall.js";

function configWith(
  conversationRecall: unknown,
  providers?: Record<string, unknown>,
): OpenClawConfig {
  return {
    models: {
      providers: providers ?? { litellm: { baseUrl: "http://127.0.0.1:4000/v1", models: [] } },
    },
    plugins: { entries: { "memory-core": { config: { conversationRecall } } } },
  } as unknown as OpenClawConfig;
}

const ENABLED = configWith({ enabled: true, provider: "litellm" });

function callWith(
  fetchImpl: typeof fetch,
  extra?: Partial<ConversationRecallCall>,
): ConversationRecallCall {
  const config = resolveConversationRecallConfig(ENABLED);
  if (!config) {
    throw new Error("config missing");
  }
  return {
    config,
    cfg: ENABLED,
    agentId: "main",
    deps: { fetch: fetchImpl, resolveApiKey: async () => "sk-test" },
    ...extra,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("resolveConversationRecallConfig", () => {
  it("is off unless explicitly enabled", () => {
    expect(resolveConversationRecallConfig(undefined)).toBeNull();
    expect(resolveConversationRecallConfig(configWith(undefined))).toBeNull();
    expect(resolveConversationRecallConfig(configWith({ enabled: false }))).toBeNull();
  });

  it("reads the endpoint from the model provider and normalizes the /v1 root", () => {
    expect(resolveConversationRecallConfig(ENABLED)).toMatchObject({
      provider: "litellm",
      root: "http://127.0.0.1:4000/v1",
    });
    const bare = configWith(
      { enabled: true, provider: "lim", timeoutMs: 5000 },
      { lim: { baseUrl: "https://proxy.example/", models: [] } },
    );
    expect(resolveConversationRecallConfig(bare)).toEqual({
      provider: "lim",
      root: "https://proxy.example/v1",
      timeoutMs: 5000,
    });
  });

  it("stays off when the provider has no base URL", () => {
    expect(
      resolveConversationRecallConfig(configWith({ enabled: true, provider: "litellm" }, {})),
    ).toBeNull();
  });

  it("stays off without an explicit provider", () => {
    expect(resolveConversationRecallConfig(configWith({ enabled: true }))).toBeNull();
    expect(
      resolveConversationRecallConfig(configWith({ enabled: true, provider: " " })),
    ).toBeNull();
  });
});

describe("resolveConversationRecallForSession", () => {
  it("allows the owner's direct sessions and refuses others, sandboxed, group, channel, and scoped runs", () => {
    const allowed = (session: Parameters<typeof resolveConversationRecallForSession>[1]) =>
      resolveConversationRecallForSession(ENABLED, { senderIsOwner: true, ...session }) !== null;
    expect(allowed({})).toBe(true);
    expect(allowed({ sessionKey: "agent:main:main" })).toBe(true);
    expect(allowed({ senderIsOwner: false })).toBe(false);
    expect(allowed({ senderIsOwner: undefined })).toBe(false);
    expect(allowed({ sessionKey: "agent:main:telegram:group:123" })).toBe(false);
    expect(allowed({ sessionKey: "agent:main:discord:channel:9" })).toBe(false);
    expect(allowed({ sandboxed: true })).toBe(false);
    expect(allowed({ scopedRecall: true })).toBe(false);
  });
});

describe("conversation paths", () => {
  it("round-trips session ids and rejects anything else", () => {
    const path = "conversation:925562bdfe9c4b29877307484b4a15d9#13";
    expect(parseConversationPath(path)).toEqual({
      sessionId: "925562bdfe9c4b29877307484b4a15d9",
      seq: 13,
    });
    expect(parseConversationPath("memory/2026-10-02.md")).toBeNull();
    expect(parseConversationPath("conversation:abc")).toBeNull();
  });
});

describe("searchStoredConversations", () => {
  it("posts the query with the key and agent id and maps hits to corpus results", async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, _init?: RequestInit) =>
      jsonResponse({
        object: "list",
        data: [
          {
            score: 0.93,
            session_id: "ses_1",
            seq: 4,
            title: "Garden plans",
            started: "2026-09-03T08:00:00+00:00",
            source: "opencode",
            agent: "sett",
            harness: "opencode",
            chunk: 0,
            content: "assistant: plant tomatoes in May",
          },
        ],
      }),
    );
    const results = await searchStoredConversations(callWith(fetchMock as typeof fetch), {
      query: "tomatoes",
      limit: 5,
    });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://127.0.0.1:4000/v1/memory/search");
    expect(JSON.parse(requestBodyText(init?.body))).toEqual({ query: "tomatoes", limit: 5 });
    expect(init?.headers).toMatchObject({
      authorization: "Bearer sk-test",
      "x-openclaw-agent-id": "main",
    });
    expect(results).toEqual([
      {
        corpus: "conversations",
        kind: "conversation",
        path: "conversation:ses_1#4",
        title: "Garden plans",
        score: 0.93,
        snippet: "assistant: plant tomatoes in May",
        id: "ses_1",
        startLine: 4,
        endLine: 4,
        source: "opencode",
        provenanceLabel: "opencode/sett",
        updatedAt: "2026-09-03T08:00:00+00:00",
      },
    ]);
  });

  it("surfaces the store's own error text and fails without a key", async () => {
    const refused = vi.fn(async () =>
      jsonResponse({ detail: "Memory search needs DATABASE_URL" }, 503),
    );
    await expect(
      searchStoredConversations(callWith(refused as unknown as typeof fetch), {
        query: "x",
        limit: 3,
      }),
    ).rejects.toThrow(/503: Memory search needs DATABASE_URL/);
    const noKey = callWith(refused as unknown as typeof fetch, {
      deps: { fetch: refused as unknown as typeof fetch, resolveApiKey: async () => undefined },
    });
    await expect(searchStoredConversations(noKey, { query: "x", limit: 3 })).rejects.toThrow(
      /no API key for provider litellm/,
    );
  });
});

describe("reranking", () => {
  it("sends trimmed documents and returns scores in input order", async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, _init?: RequestInit) =>
      jsonResponse({
        data: [
          { index: 1, score: 0.9 },
          { index: 0, score: 0.1 },
        ],
      }),
    );
    const notes = { snippet: "notes" };
    const long = { snippet: "x".repeat(9000) };
    const { ranked, scores } = await rankByStoreRerank(callWith(fetchMock as typeof fetch), {
      query: "q",
      candidates: [notes, long],
      limit: 2,
    });
    expect([scores.get(notes), scores.get(long)]).toEqual([0.1, 0.9]);
    expect(ranked).toEqual([long, notes]);
    const body = JSON.parse(requestBodyText(fetchMock.mock.calls[0]![1]?.body)) as {
      documents: string[];
    };
    expect(fetchMock.mock.calls[0]![0]).toBe("http://127.0.0.1:4000/v1/memory/rerank");
    expect(body.documents[1]!.length).toBeLessThan(4100);
  });

  it("orders mixed candidates by score and leaves unscored ones last", async () => {
    const candidates = [
      { snippet: "note a" },
      { snippet: "conversation b" },
      { snippet: "note c" },
    ];
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        data: [
          { index: 0, score: 0.2 },
          { index: 1, score: 0.8 },
        ],
      }),
    );
    const { ranked, scores } = await rankByStoreRerank(
      callWith(fetchMock as unknown as typeof fetch),
      {
        query: "q",
        candidates,
        limit: 2,
      },
    );
    expect(ranked).toEqual([candidates[1], candidates[0]]);
    expect(scores.get(candidates[1]!)).toBe(0.8);
    expect(scores.has(candidates[2]!)).toBe(false);
  });
});

describe("readStoredConversation", () => {
  it("windows the read around the hit and formats the messages", async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, _init?: RequestInit) =>
      jsonResponse({
        object: "memory.context",
        session_id: "ses_1",
        title: "Garden plans",
        started: "2026-09-03T08:00:00+00:00",
        harness: "opencode",
        agent: "sett",
        last_seq: 9,
        items: [
          { seq: 3, text: "user: when?", hit: false },
          { seq: 4, text: "assistant: May", hit: true },
        ],
      }),
    );
    const read = await readStoredConversation(callWith(fetchMock as typeof fetch), {
      sessionId: "ses_1",
      seq: 4,
      lines: 5,
    });
    expect(JSON.parse(requestBodyText(fetchMock.mock.calls[0]![1]?.body))).toEqual({
      session_id: "ses_1",
      seq: 4,
      before: 2,
      after: 2,
    });
    expect(read).toMatchObject({
      path: "conversation:ses_1#4",
      title: "Garden plans",
      fromSeq: 3,
      toSeq: 4,
      lastSeq: 9,
      harness: "opencode",
    });
    expect(read?.text).toBe("   seq=3 user: when?\n>> seq=4 assistant: May");
  });

  it("answers null for a session outside the caller's memory and throws other failures", async () => {
    const missing = vi.fn(async () =>
      jsonResponse({ detail: "No session ses_x in your memory." }, 404),
    );
    await expect(
      readStoredConversation(callWith(missing as unknown as typeof fetch), {
        sessionId: "ses_x",
        seq: 0,
      }),
    ).resolves.toBeNull();
    const broken = vi.fn(async () => jsonResponse({ detail: "boom" }, 500));
    await expect(
      readStoredConversation(callWith(broken as unknown as typeof fetch), {
        sessionId: "ses_x",
        seq: 0,
      }),
    ).rejects.toMatchObject({ name: "ConversationRecallError", status: 500 });
  });
});
