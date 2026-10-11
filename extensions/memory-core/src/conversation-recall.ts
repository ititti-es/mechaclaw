import type {
  MemoryCorpusSearchResult,
  OpenClawConfig,
} from "openclaw/plugin-sdk/memory-core-host-runtime-core";
import { parseAgentSessionKey } from "openclaw/plugin-sdk/memory-core-host-runtime-core";
import { isRecord } from "openclaw/plugin-sdk/string-coerce-runtime";

/**
 * Conversation recall over a LiteLLM-compatible proxy's session store, blended into `memory_search` / `memory_get`.
 *
 * The proxy owns every conversation (OpenClaw, opencode, Claude Code) and answers `POST /v1/memory/search`
 * with embedding + rerank, scoped to the calling key's user. This module is only the client: it never sees
 * the store, and a missing or failing store degrades `memory_search` to the workspace notes with a warning.
 *
 * Enabled per instance with `plugins.entries.memory-core.config.conversationRecall.enabled=true` and an explicit
 * `provider`; the endpoint and key come from that model provider, so no extra secret is introduced.
 */

const DEFAULT_TIMEOUT_MS = 20_000;
const MAX_SEARCH_LIMIT = 50;
const MAX_RERANK_DOCUMENTS = 64;
const RERANK_DOCUMENT_CHARS = 4000;
const SNIPPET_CHARS = 1500;
const MAX_CONTEXT_SIDE = 20;
const DEFAULT_CONTEXT_SIDE = 4;

const CONVERSATION_PATH_PREFIX = "conversation:";
const CONVERSATION_PATH_PATTERN = /^conversation:([^#\s]+)#(\d+)$/;

export type ConversationRecallConfig = {
  provider: string;
  /** OpenAI-compatible root of the provider, e.g. `http://127.0.0.1:4000/v1`. */
  root: string;
  timeoutMs: number;
};

type ConversationRecallDeps = {
  fetch?: typeof fetch;
  resolveApiKey?: (params: {
    provider: string;
    cfg: OpenClawConfig;
  }) => Promise<string | undefined>;
};

export type ConversationRecallCall = {
  config: ConversationRecallConfig;
  cfg: OpenClawConfig;
  agentId?: string;
  signal?: AbortSignal;
  deps?: ConversationRecallDeps;
};

type StoredConversationHit = {
  score: number;
  session_id: string;
  seq: number;
  title: string | null;
  source: string | null;
  agent: string | null;
  harness: string | null;
  started?: string | null;
  content: string;
};

type StoredConversationContext = {
  session_id: string;
  title: string | null;
  started: string | null;
  harness: string | null;
  agent: string | null;
  last_seq: number | null;
  items: Array<{ seq: number; text: string; hit: boolean }>;
};

export type ConversationRead = {
  path: string;
  title?: string;
  text: string;
  fromSeq?: number;
  toSeq?: number;
  lastSeq?: number;
  started?: string;
  harness?: string;
  agent?: string;
};

class ConversationRecallError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ConversationRecallError";
  }
}

/** The conversationRecall block of the memory-core plugin config, or null when recall is not enabled. */
export function resolveConversationRecallConfig(
  cfg: OpenClawConfig | undefined,
): ConversationRecallConfig | null {
  const plugin: unknown = cfg?.plugins?.entries?.["memory-core"]?.config;
  const raw = isRecord(plugin) ? plugin.conversationRecall : undefined;
  if (!isRecord(raw) || raw.enabled !== true) {
    return null;
  }
  const provider = typeof raw.provider === "string" ? raw.provider.trim() : "";
  if (!provider) {
    return null;
  }
  const baseUrl: unknown = cfg?.models?.providers?.[provider]?.baseUrl;
  if (typeof baseUrl !== "string" || !baseUrl.trim()) {
    return null;
  }
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  const timeoutMs =
    typeof raw.timeoutMs === "number" && raw.timeoutMs > 0 ? raw.timeoutMs : DEFAULT_TIMEOUT_MS;
  return { provider, root: trimmed.endsWith("/v1") ? trimmed : `${trimmed}/v1`, timeoutMs };
}

/**
 * Recall spans every private conversation the user has had, so it is limited to turns the runtime marks as
 * the owner's, in direct, unsandboxed sessions without a trusted-runtime recall scope. Everyone else (other
 * senders, group and channel sessions) keeps the workspace notes only.
 */
export function resolveConversationRecallForSession(
  cfg: OpenClawConfig | undefined,
  session: {
    sessionKey?: string;
    sandboxed?: boolean;
    scopedRecall?: boolean;
    senderIsOwner?: boolean;
  },
): ConversationRecallConfig | null {
  if (
    session.senderIsOwner !== true ||
    session.sandboxed === true ||
    session.scopedRecall === true
  ) {
    return null;
  }
  const rest = parseAgentSessionKey(session.sessionKey)?.rest?.toLowerCase();
  if (rest && /(^|:)(group|channel)(:|$)/.test(rest)) {
    return null;
  }
  return resolveConversationRecallConfig(cfg);
}

async function defaultResolveApiKey(params: {
  provider: string;
  cfg: OpenClawConfig;
}): Promise<string | undefined> {
  const { resolveApiKeyForProvider } = await import("openclaw/plugin-sdk/provider-auth-runtime");
  return (await resolveApiKeyForProvider({ provider: params.provider, cfg: params.cfg }))?.apiKey;
}

async function post<T>(call: ConversationRecallCall, path: string, body: unknown): Promise<T> {
  const { config, cfg, agentId, signal, deps } = call;
  const apiKey = await (deps?.resolveApiKey ?? defaultResolveApiKey)({
    provider: config.provider,
    cfg,
  });
  if (!apiKey) {
    throw new ConversationRecallError(`no API key for provider ${config.provider}`);
  }
  const timeout = AbortSignal.timeout(config.timeoutMs);
  const response = await (deps?.fetch ?? fetch)(`${config.root}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${apiKey}`,
      "content-type": "application/json",
      ...(agentId ? { "x-openclaw-agent-id": agentId } : {}),
    },
    body: JSON.stringify(body),
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    let message = detail.slice(0, 200);
    try {
      const parsed: unknown = JSON.parse(detail);
      if (isRecord(parsed) && typeof parsed.detail === "string") {
        message = parsed.detail.slice(0, 200);
      }
    } catch {
      // Keep the raw text.
    }
    throw new ConversationRecallError(
      `session store ${path} answered ${response.status}: ${message}`,
      response.status,
    );
  }
  return (await response.json()) as T;
}

function clip(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit).trimEnd()} [...]`;
}

function conversationPath(sessionId: string, seq: number): string {
  return `${CONVERSATION_PATH_PREFIX}${sessionId}#${seq}`;
}

export function parseConversationPath(lookup: string): { sessionId: string; seq: number } | null {
  const match = CONVERSATION_PATH_PATTERN.exec(lookup.trim());
  return match ? { sessionId: match[1]!, seq: Number(match[2]) } : null;
}

/** Reranked conversation passages for a query, shaped like the other memory corpora's results. */
export async function searchStoredConversations(
  call: ConversationRecallCall,
  params: { query: string; limit: number },
): Promise<MemoryCorpusSearchResult[]> {
  const limit = Math.max(1, Math.min(MAX_SEARCH_LIMIT, params.limit));
  const response = await post<{ data?: StoredConversationHit[] }>(call, "/memory/search", {
    query: params.query,
    limit,
  });
  return (response.data ?? []).map((hit) => {
    const where = [hit.harness, hit.agent].filter(Boolean).join("/");
    return {
      corpus: "conversations",
      kind: "conversation",
      path: conversationPath(hit.session_id, hit.seq),
      title: hit.title ?? undefined,
      score: hit.score,
      snippet: clip(hit.content.trim(), SNIPPET_CHARS),
      id: hit.session_id,
      startLine: hit.seq,
      endLine: hit.seq,
      source: hit.harness ?? undefined,
      provenanceLabel: where || undefined,
      updatedAt: hit.started ?? undefined,
    } satisfies MemoryCorpusSearchResult;
  });
}

/** Reranker scores for the candidates, in input order (higher is more relevant). */
async function rerankWithStore(
  call: ConversationRecallCall,
  params: { query: string; documents: readonly string[] },
): Promise<number[]> {
  const documents = params.documents
    .slice(0, MAX_RERANK_DOCUMENTS)
    .map((document) => clip(document.trim() || "(empty)", RERANK_DOCUMENT_CHARS));
  if (documents.length === 0) {
    return [];
  }
  const response = await post<{ data?: Array<{ index: number; score: number }> }>(
    call,
    "/memory/rerank",
    { query: params.query, documents },
  );
  const scores = Array.from({ length: documents.length }, () => Number.NEGATIVE_INFINITY);
  for (const entry of response.data ?? []) {
    if (entry.index >= 0 && entry.index < scores.length) {
      scores[entry.index] = entry.score;
    }
  }
  return scores;
}

/** The messages around a conversation hit; `lines` sizes the window around the hit message. */
export async function readStoredConversation(
  call: ConversationRecallCall,
  params: { sessionId: string; seq: number; lines?: number },
): Promise<ConversationRead | null> {
  const span = Math.max(
    1,
    Math.min(2 * MAX_CONTEXT_SIDE + 1, params.lines ?? 2 * DEFAULT_CONTEXT_SIDE + 1),
  );
  const before = Math.min(MAX_CONTEXT_SIDE, Math.floor((span - 1) / 2));
  const after = Math.min(MAX_CONTEXT_SIDE, span - 1 - before);
  let context: StoredConversationContext;
  try {
    context = await post<StoredConversationContext>(call, "/memory/context", {
      session_id: params.sessionId,
      seq: params.seq,
      before,
      after,
    });
  } catch (error) {
    if (error instanceof ConversationRecallError && error.status === 404) {
      return null;
    }
    throw error;
  }
  const items = context.items ?? [];
  return {
    path: conversationPath(context.session_id, params.seq),
    title: context.title ?? undefined,
    text: items.map((item) => `${item.hit ? ">>" : "  "} seq=${item.seq} ${item.text}`).join("\n"),
    fromSeq: items[0]?.seq,
    toSeq: items.at(-1)?.seq,
    lastSeq: context.last_seq ?? undefined,
    started: context.started ?? undefined,
    harness: context.harness ?? undefined,
    agent: context.agent ?? undefined,
  };
}

/**
 * Orders mixed candidates (workspace notes and conversation hits) by one reranker, so their scores are
 * comparable. Candidates the reranker did not score sink to the end in their original order.
 */
export async function rankByStoreRerank<T extends { snippet: string }>(
  call: ConversationRecallCall,
  params: { query: string; candidates: readonly T[]; limit: number },
): Promise<{ ranked: T[]; scores: Map<T, number> }> {
  const scored = params.candidates.slice(0, MAX_RERANK_DOCUMENTS);
  const values = await rerankWithStore(call, {
    query: params.query,
    documents: scored.map((candidate) => candidate.snippet),
  });
  const scores = new Map<T, number>();
  scored.forEach((candidate, index) => {
    const score = values[index];
    if (score !== undefined && Number.isFinite(score)) {
      scores.set(candidate, score);
    }
  });
  const ranked = [...scored]
    .map((candidate, index) => ({ candidate, index, score: scores.get(candidate) }))
    .toSorted(
      (left, right) =>
        (right.score ?? Number.NEGATIVE_INFINITY) - (left.score ?? Number.NEGATIVE_INFINITY) ||
        left.index - right.index,
    )
    .slice(0, Math.max(1, params.limit))
    .map((entry) => entry.candidate);
  return { ranked, scores };
}
