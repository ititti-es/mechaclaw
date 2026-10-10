import { describe, expect, it } from "vitest";
import {
  INTERNAL_RUNTIME_CONTEXT_BEGIN,
  INTERNAL_RUNTIME_CONTEXT_END,
} from "../../agents/internal-runtime-context.js";
import type { OpenClawConfig } from "../../config/types.openclaw.js";
import { storedTurnsByResponse } from "./chat-history-server-owned-turns.js";
import {
  overlayServerOwnedContent,
  resolveServerOwnedHistoryRoute,
} from "./chat-history-server-owned.js";

const META = "__openclaw";
const carrier = `${INTERNAL_RUNTIME_CONTEXT_BEGIN}\nnow: monday\n${INTERNAL_RUNTIME_CONTEXT_END}`;

function userInput(text: string) {
  return { type: "message", role: "user", content: [{ type: "input_text", text }] };
}

function assistantOutput(text: string) {
  return { type: "message", role: "assistant", content: [{ type: "output_text", text }] };
}

const stored = storedTurnsByResponse({
  items: [
    { seq: 0, is_output: false, content: userInput("[Thu 2026-10-01 17:40 CDT] server question") },
    { seq: 1, is_output: false, content: userInput(carrier) },
    {
      seq: 2,
      is_output: true,
      content: {
        type: "function_call",
        call_id: "call_1",
        name: "lookup",
        arguments: '{"q":"server"}',
      },
    },
    {
      seq: 3,
      is_output: false,
      content: { type: "function_call_output", call_id: "call_1", output: "server tool output" },
    },
    { seq: 4, is_output: true, content: assistantOutput("server answer") },
    { seq: 5, is_output: false, content: userInput(`${carrier}\n\nsecond server question`) },
    { seq: 6, is_output: true, content: assistantOutput("second server answer") },
  ],
  responses: [
    { id: "resp_2", item_count: 5 },
    { id: "resp_1", item_count: 3 },
    { id: "resp_3", item_count: 7 },
  ],
});

const local = [
  { role: "user", content: "local question", __openclaw: { id: "u1", idempotencyKey: "k1" } },
  {
    role: "assistant",
    responseId: "resp_1",
    content: [{ type: "toolCall", id: "call_1|fc_1", name: "lookup", arguments: { q: "local" } }],
    __openclaw: { id: "a1", runId: "run_1" },
  },
  {
    role: "toolResult",
    toolCallId: "call_1|fc_1",
    content: [{ type: "text", text: "local tool output" }],
    __openclaw: { id: "t1", runId: "run_1" },
  },
  { role: "custom", customType: "card", content: [], __openclaw: { id: "c1" } },
  {
    role: "assistant",
    responseId: "resp_2",
    content: [{ type: "text", text: "local answer" }],
    __openclaw: { id: "a2", runId: "run_1" },
  },
  {
    role: "user",
    content: [{ type: "text", text: "second local question" }],
    __openclaw: { id: "u2" },
  },
  {
    role: "assistant",
    responseId: "resp_3",
    content: [
      { type: "thinking", thinking: "local thought" },
      { type: "text", text: "x" },
    ],
    __openclaw: { id: "a3" },
  },
  { role: "user", content: "pending local question", __openclaw: { id: "u3" } },
];

describe("server-owned chat history", () => {
  it("splits the stored transcript at each response boundary", () => {
    expect([...stored.keys()]).toEqual(["resp_1", "resp_2", "resp_3"]);
    expect(stored.get("resp_1")?.inputs).toHaveLength(2);
    expect(stored.get("resp_1")?.outputs).toHaveLength(1);
    expect(stored.get("resp_2")?.inputs).toHaveLength(1);
  });

  it("takes content from the store and keeps local metadata and local-only messages", () => {
    const { messages, replaced } = overlayServerOwnedContent(local, stored);
    const [user, call, tool, card, answer, user2, answer2, pending] = messages as Array<
      Record<string, any>
    >;

    expect(user?.content).toBe("server question");
    expect(user?.[META]).toEqual({ id: "u1", idempotencyKey: "k1", contentSource: "server" });
    expect(call?.content).toEqual([
      { type: "toolCall", id: "call_1|fc_1", name: "lookup", arguments: { q: "server" } },
    ]);
    expect(tool?.content).toEqual([{ type: "text", text: "server tool output" }]);
    expect(tool?.toolCallId).toBe("call_1|fc_1");
    expect(card).toBe(local[3]);
    expect(answer?.content).toEqual([{ type: "text", text: "server answer" }]);
    expect(answer?.[META].runId).toBe("run_1");
    expect(user2?.content).toEqual([{ type: "text", text: "second server question" }]);
    expect(answer2?.content).toEqual([
      { type: "thinking", thinking: "local thought" },
      { type: "text", text: "second server answer" },
    ]);
    expect(pending).toBe(local[7]);
    expect(replaced).toBe(6);
  });

  it("reads each response's turn from its own compaction version and skips summaries", () => {
    const turns = storedTurnsByResponse({
      items: [
        { version: 1, seq: 0, is_output: false, content: userInput("first") },
        { version: 1, seq: 1, is_output: true, content: assistantOutput("one") },
        { version: 1, seq: 2, is_output: false, content: userInput("second") },
        { version: 1, seq: 3, is_output: true, content: assistantOutput("two") },
        { version: 2, seq: 0, is_output: false, summary: true, content: userInput("summary") },
        { version: 2, seq: 1, is_output: false, content: userInput("second") },
        { version: 2, seq: 2, is_output: true, content: assistantOutput("two") },
      ],
      responses: [
        { id: "resp_1", version: 1, item_count: 2 },
        { id: "resp_2", version: 2, item_count: 3 },
      ],
    });

    expect(turns.get("resp_1")?.inputs).toEqual([userInput("first")]);
    expect(turns.get("resp_2")?.inputs).toEqual([userInput("second")]);
    expect(turns.get("resp_2")?.outputs).toEqual([assistantOutput("two")]);
  });

  it("keys turns by the advertised id the store matched", () => {
    const turns = storedTurnsByResponse({
      items: [
        { seq: 0, is_output: false, content: userInput("q") },
        { seq: 1, is_output: true, content: assistantOutput("a") },
      ],
      responses: [{ id: "provider-id", client_id: "resp_advertised", item_count: 2 }],
    });

    expect([...turns.keys()]).toEqual(["resp_advertised"]);
  });

  it("keeps local content when a turn cannot be joined exactly", () => {
    const { messages, replaced } = overlayServerOwnedContent(
      [
        { role: "user", content: "one" },
        { role: "user", content: "two" },
        {
          role: "assistant",
          responseId: "resp_3",
          content: [
            { type: "text", text: "a" },
            { type: "text", text: "b" },
          ],
        },
        { role: "assistant", responseId: "resp_missing", content: [{ type: "text", text: "c" }] },
      ],
      stored,
    );

    expect((messages[0] as { content: string }).content).toBe("one");
    expect((messages[2] as { content: unknown[] }).content).toEqual([
      { type: "text", text: "a" },
      { type: "text", text: "b" },
    ]);
    expect(replaced).toBe(0);
  });

  it("only routes models that opt into server-owned history", () => {
    const cfg = {
      models: {
        providers: {
          litellm: {
            baseUrl: "http://127.0.0.1:4000/v1",
            models: [
              {
                id: "owned",
                compat: {
                  supportsResponsesContinuation: true,
                  responsesHistoryOwnedByServer: true,
                },
              },
              { id: "stateless", compat: { supportsResponsesContinuation: true } },
              {
                id: "pruned",
                compat: {
                  supportsResponsesContinuation: true,
                  responsesHistoryOwnedByServer: true,
                  responsesPruneLocalContent: true,
                },
              },
            ],
          },
        },
      },
    } as unknown as OpenClawConfig;

    expect(resolveServerOwnedHistoryRoute(cfg, "litellm", "owned")).toEqual({
      provider: "litellm",
      modelId: "owned",
      baseUrl: "http://127.0.0.1:4000/v1",
      pruneLocalContent: false,
    });
    expect(resolveServerOwnedHistoryRoute(cfg, "litellm", "pruned")?.pruneLocalContent).toBe(true);
    expect(resolveServerOwnedHistoryRoute(cfg, "litellm", "stateless")).toBeUndefined();
    expect(resolveServerOwnedHistoryRoute(cfg, "other", "owned")).toBeUndefined();
  });
});
