// Memory Core tests cover memory_search and memory_get with liminal conversation recall.
import { clearMemoryPluginState } from "openclaw/plugin-sdk/memory-host-core";
import { requestBodyText, requestUrl } from "openclaw/plugin-sdk/test-env";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetMemoryToolMockState, setMemorySearchImpl } from "./memory-tool-manager.test-mocks.js";
import { createMemoryGetTool, createMemorySearchTool } from "./tools.js";
import { asOpenClawConfig } from "./tools.test-helpers.js";

vi.mock("openclaw/plugin-sdk/provider-auth-runtime", () => ({
  resolveApiKeyForProvider: async () => ({ apiKey: "sk-test", source: "test", mode: "api-key" }),
}));

const NOTE = {
  path: "MEMORY.md",
  startLine: 3,
  endLine: 4,
  score: 0.7,
  snippet: "Voice notes: clone from reference sample.",
  source: "memory" as const,
};

const HIT = {
  score: 0.93,
  session_id: "ses_1",
  seq: 13,
  title: "Fish voice cloning",
  started: "2026-09-28T12:00:00+00:00",
  source: "opencode",
  agent: "example-agent",
  harness: "opencode",
  chunk: 0,
  content: "assistant: Fish S2.1 Pro does instant voice cloning from a reference clip.",
};

function config(liminal?: unknown) {
  return asOpenClawConfig({
    agents: { list: [{ id: "main", default: true }] },
    models: { providers: { liminal: { baseUrl: "http://127.0.0.1:4000/v1", models: [] } } },
    plugins: { entries: { "memory-core": { config: liminal ? { liminal } : {} } } },
  } as never);
}

type Routes = Record<string, (body: Record<string, unknown>) => Response>;

function stubLiminal(routes: Routes) {
  const calls: Array<{ path: string; body: Record<string, unknown> }> = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const path = new URL(requestUrl(url)).pathname.replace("/v1", "");
      const body = JSON.parse(requestBodyText(init?.body)) as Record<string, unknown>;
      calls.push({ path, body });
      const route = routes[path];
      return route ? route(body) : new Response("{}", { status: 404 });
    }),
  );
  return calls;
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

async function search(
  cfg: ReturnType<typeof config>,
  params: Record<string, unknown>,
  options = {},
) {
  const tool = createMemorySearchTool({
    config: cfg,
    agentSessionKey: "agent:main:main",
    senderIsOwner: true,
    ...options,
  });
  if (!tool) {
    throw new Error("tool missing");
  }
  const result = await tool.execute("call", params);
  return result.details as {
    results: Array<{ path: string; score: number; corpus?: string; snippet: string }>;
    warning?: string;
    conversations?: { outcome: string; count: number; reranked: boolean };
  };
}

describe("memory_search with liminal conversation recall", () => {
  beforeEach(() => {
    clearMemoryPluginState();
    resetMemoryToolMockState();
    setMemorySearchImpl(async () => [NOTE]);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("ranks notes and conversation hits together on the reranker's scores", async () => {
    const calls = stubLiminal({
      "/memory/search": () => json({ object: "list", data: [HIT] }),
      "/memory/rerank": () =>
        json({
          data: [
            { index: 0, score: 0.2 },
            { index: 1, score: 0.97 },
          ],
        }),
    });
    const details = await search(config({ enabled: true }), {
      query: "which voice model for the agent",
    });
    expect(details.results.map((result) => result.path)).toEqual([
      "conversation:ses_1#13",
      "MEMORY.md",
    ]);
    expect(details.results.map((result) => result.score)).toEqual([0.97, 0.2]);
    expect(details.results[0]?.corpus).toBe("conversations");
    expect(details.conversations).toEqual({ outcome: "ok", count: 1, reranked: true });
    expect(calls.map((call) => call.path).toSorted()).toEqual(["/memory/rerank", "/memory/search"]);
    expect(
      (
        calls.find((call) => call.path === "/memory/rerank")?.body.documents as string[] | undefined
      )?.[0],
    ).toContain("Voice notes: clone from reference sample.");
  });

  it("falls back to the notes with a warning when liminal is down", async () => {
    stubLiminal({ "/memory/search": () => json({ detail: "down" }, 503) });
    const details = await search(config({ enabled: true }), { query: "voice" });
    expect(details.results.map((result) => result.path)).toEqual(["MEMORY.md"]);
    expect(details.conversations).toMatchObject({ outcome: "unavailable", count: 0 });
    expect(details.warning).toContain("Conversation recall is unavailable");
  });

  it("keeps both streams when only the reranker fails, and says the order is approximate", async () => {
    stubLiminal({
      "/memory/search": () => json({ object: "list", data: [HIT] }),
      "/memory/rerank": () => json({ detail: "rerank down" }, 502),
    });
    const details = await search(config({ enabled: true }), { query: "voice" });
    expect(details.results.map((result) => result.path)).toEqual([
      "conversation:ses_1#13",
      "MEMORY.md",
    ]);
    expect(details.conversations?.reranked).toBe(false);
    expect(details.warning).toContain("not reranked");
  });

  it("serves the sessions corpus from liminal only, and the memory corpus from the notes only", async () => {
    const calls = stubLiminal({
      "/memory/search": () => json({ object: "list", data: [HIT] }),
      "/memory/rerank": () => json({ data: [{ index: 0, score: 0.9 }] }),
    });
    const sessions = await search(config({ enabled: true }), {
      query: "voice",
      corpus: "sessions",
    });
    expect(sessions.results.map((result) => result.path)).toEqual(["conversation:ses_1#13"]);
    calls.length = 0;
    const memory = await search(config({ enabled: true }), { query: "voice", corpus: "memory" });
    expect(memory.results.map((result) => result.path)).toEqual(["MEMORY.md"]);
    expect(calls).toEqual([]);
  });

  it("never reaches liminal when disabled, for other senders, sandboxed, or in a group", async () => {
    const calls = stubLiminal({ "/memory/search": () => json({ object: "list", data: [HIT] }) });
    for (const [cfg, options] of [
      [config(), {}],
      [config({ enabled: true }), { senderIsOwner: false }],
      [config({ enabled: true }), { senderIsOwner: undefined }],
      [config({ enabled: true }), { sandboxed: true }],
      [config({ enabled: true }), { agentSessionKey: "agent:main:telegram:group:42" }],
    ] as const) {
      const details = await search(cfg, { query: "voice" }, options);
      expect(details.results.map((result) => result.path)).toEqual(["MEMORY.md"]);
    }
    expect(calls).toEqual([]);
  });
});

describe("memory_get for conversation hits", () => {
  beforeEach(() => {
    clearMemoryPluginState();
    resetMemoryToolMockState();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reads the messages around a hit from liminal", async () => {
    const calls = stubLiminal({
      "/memory/context": () =>
        json({
          session_id: "ses_1",
          title: "Fish voice cloning",
          started: "2026-09-28T12:00:00+00:00",
          harness: "opencode",
          agent: "example-agent",
          last_seq: 31,
          items: [
            { seq: 12, text: "tool output: Fish docs", hit: false },
            { seq: 13, text: "assistant: Found it", hit: true },
          ],
        }),
    });
    const tool = createMemoryGetTool({
      config: config({ enabled: true }),
      agentSessionKey: "agent:main:main",
      senderIsOwner: true,
    });
    const result = await tool!.execute("call", { path: "conversation:ses_1#13", lines: 3 });
    expect(calls[0]).toEqual({
      path: "/memory/context",
      body: { session_id: "ses_1", seq: 13, before: 1, after: 1 },
    });
    expect(result.details).toMatchObject({
      status: "ok",
      corpus: "conversations",
      path: "conversation:ses_1#13",
      title: "Fish voice cloning",
      text: "   seq=12 tool output: Fish docs\n>> seq=13 assistant: Found it",
    });
  });

  it("reports not_found for a session liminal will not show this caller", async () => {
    stubLiminal({ "/memory/context": () => json({ detail: "No session in your memory." }, 404) });
    const tool = createMemoryGetTool({
      config: config({ enabled: true }),
      agentSessionKey: "agent:main:main",
      senderIsOwner: true,
    });
    const result = await tool!.execute("call", { path: "conversation:ses_x#0" });
    expect(result.details).toMatchObject({ status: "not_found", corpus: "conversations" });
  });
});
