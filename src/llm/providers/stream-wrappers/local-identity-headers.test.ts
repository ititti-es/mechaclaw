// Identity headers reach loopback gateways only, and never override caller headers.
import type { StreamFn } from "openclaw/plugin-sdk/agent-core";
import type { Model } from "openclaw/plugin-sdk/llm";
import { createAssistantMessageEventStream } from "openclaw/plugin-sdk/llm";
import { describe, expect, it } from "vitest";
import { createLocalEndpointIdentityHeadersWrapper } from "./local-identity-headers.js";

function headersSent(
  baseUrl: string,
  agentId: string | undefined,
  callerHeaders?: Record<string, string>,
): Record<string, string> | undefined {
  const calls: Array<Record<string, string> | undefined> = [];
  const baseStreamFn: StreamFn = (_model, _context, options) => {
    calls.push(options?.headers);
    return createAssistantMessageEventStream();
  };
  const wrapped = createLocalEndpointIdentityHeadersWrapper(baseStreamFn, agentId);
  void wrapped(
    {
      api: "openai-responses",
      provider: "liminal",
      id: "composer-2.5",
      baseUrl,
    } as Model<"openai-responses">,
    { messages: [] },
    callerHeaders ? { headers: callerHeaders } : {},
  );
  return calls[0];
}

describe("local endpoint identity headers", () => {
  it("names the agent and harness to a loopback gateway", () => {
    expect(headersSent("http://127.0.0.1:4000/v1", "example-agent")).toEqual({
      "x-litellm-harness": "openclaw",
      "x-openclaw-agent-id": "example-agent",
    });
  });

  it("sends only the harness when the agent is unknown", () => {
    expect(headersSent("http://localhost:4000/v1", undefined)).toEqual({
      "x-litellm-harness": "openclaw",
    });
  });

  it("never sends them to a remote endpoint", () => {
    expect(headersSent("https://api.example.com/v1", "example-agent")).toBeUndefined();
  });

  it("lets caller headers win", () => {
    expect(
      headersSent("http://127.0.0.1:4000/v1", "example-agent", { "x-openclaw-agent-id": "other" }),
    ).toEqual({
      "x-litellm-harness": "openclaw",
      "x-openclaw-agent-id": "other",
    });
  });
});
