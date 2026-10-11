import { formatErrorMessage } from "openclaw/plugin-sdk/error-runtime";
import {
  jsonResult,
  type MemoryCorpusSearchResult,
  type OpenClawConfig,
} from "openclaw/plugin-sdk/memory-core-host-runtime-core";
import {
  parseConversationPath,
  rankByStoreRerank,
  readStoredConversation,
  resolveConversationRecallForSession,
  searchStoredConversations,
  type ConversationRecallCall,
  type ConversationRecallConfig,
} from "./conversation-recall.js";
import { runMemoryCorpusDeadline } from "./memory-corpus.js";
import type { MemoryToolOptions } from "./memory-tool-contract.js";
import type { MemorySearchToolResult } from "./tools.citations.js";

/**
 * How `memory_search` / `memory_get` use conversation recall (see conversation-recall.ts for the client).
 * Kept out of tools.ts so the search tool only decides when to ask and the rules for blending live here.
 */

export type ConversationAttempt = { results: MemoryCorpusSearchResult[]; error?: string };

type ConversationCall = Omit<ConversationRecallCall, "config">;

export type ConversationRecall = {
  /** Notes and conversation hits in final order, at most `limit`. */
  results: MemorySearchToolResult[];
  warnings: string[];
  summary?: { outcome: "ok" | "unavailable"; count: number; reranked: boolean };
  /** Puts the reranker's score on a result the model will see, leaving untouched results as they were. */
  rescore: (
    original: MemorySearchToolResult,
    shown: MemorySearchToolResult,
  ) => MemorySearchToolResult;
};

/** The recall config this tool call may use, or null (disabled, not the owner, sandboxed, group, scoped). */
export function resolveConversationRecall(
  cfg: OpenClawConfig,
  options: Pick<
    MemoryToolOptions,
    "agentSessionKey" | "sandboxed" | "conversationRecall" | "senderIsOwner"
  >,
): ConversationRecallConfig | null {
  return resolveConversationRecallForSession(cfg, {
    sessionKey: options.agentSessionKey,
    sandboxed: options.sandboxed,
    scopedRecall: options.conversationRecall !== undefined,
    senderIsOwner: options.senderIsOwner,
  });
}

/** Conversations are searched by default and for `all` and `sessions`; `memory` and `wiki` exclude them. */
export function searchesConversationCorpus(
  recall: ConversationRecallConfig | null,
  corpus: "memory" | "wiki" | "all" | "sessions" | undefined,
): recall is ConversationRecallConfig {
  return recall !== null && (corpus === undefined || corpus === "all" || corpus === "sessions");
}

/** A failing session store never fails the search: it degrades to the workspace notes with a warning. */
export async function attemptConversationSearch(
  call: ConversationRecallCall,
  query: string,
  limit: number,
): Promise<ConversationAttempt> {
  try {
    return { results: await searchStoredConversations(call, { query, limit }) };
  } catch (error) {
    if (call.signal?.aborted) {
      throw error;
    }
    return { results: [], error: formatErrorMessage(error) };
  }
}

/**
 * Ranks notes and conversation hits on one reranker scale. If the reranker is unreachable the two streams
 * are interleaved by their own scores instead, and the model is told the order is approximate.
 */
export async function finishConversationRecall(params: {
  attempt: ConversationAttempt | null;
  config: ConversationRecallConfig | null;
  call: ConversationCall;
  query: string;
  notes: MemorySearchToolResult[];
  limit: number;
  /** Set when the workspace notes search failed but conversation hits can still answer. */
  notesError?: string;
  interleave: (
    notes: MemorySearchToolResult[],
    conversations: MemorySearchToolResult[],
  ) => MemorySearchToolResult[];
}): Promise<ConversationRecall> {
  const { attempt, config } = params;
  const unchanged = (): ConversationRecall => ({
    results: params.notes,
    warnings: [],
    rescore: (_original, shown) => shown,
  });
  if (!attempt || !config) {
    return unchanged();
  }
  const warnings = [
    ...(attempt.error
      ? [
          `Conversation recall is unavailable, so only workspace notes were searched: ${attempt.error}`,
        ]
      : []),
    ...(params.notesError && attempt.results.length > 0
      ? [`Workspace notes search is unavailable: ${params.notesError}`]
      : []),
  ];
  const summary = {
    outcome: attempt.error ? ("unavailable" as const) : ("ok" as const),
    count: attempt.results.length,
    reranked: false,
  };
  if (attempt.results.length === 0) {
    return { ...unchanged(), warnings, summary };
  }
  const call = { ...params.call, config };
  const scores = new Map<MemorySearchToolResult, number>();
  let results: MemorySearchToolResult[];
  try {
    const ranked = await rankByStoreRerank(call, {
      query: params.query,
      candidates: [...params.notes, ...attempt.results] as MemorySearchToolResult[],
      limit: params.limit,
    });
    results = ranked.ranked;
    ranked.scores.forEach((score, result) => scores.set(result, score));
  } catch (error) {
    if (call.signal?.aborted) {
      throw error;
    }
    results = params.interleave(params.notes, attempt.results).slice(0, params.limit);
    warnings.push(
      `Conversation hits were not reranked, so their order is approximate: ${formatErrorMessage(error)}`,
    );
  }
  return {
    results,
    warnings,
    summary: { ...summary, reranked: scores.size > 0 },
    rescore: (original, shown) => {
      const score = scores.get(original);
      return score === undefined ? shown : Object.assign({}, shown, { score });
    },
  };
}

/** Answers a `memory_get` for a `conversation:<session>#<seq>` path, or null when it is not one. */
export async function executeConversationRead(params: {
  cfg: OpenClawConfig;
  agentId: string;
  options: MemoryToolOptions;
  relPath: string;
  lines?: number;
  signal?: AbortSignal;
}) {
  const target = parseConversationPath(params.relPath);
  const config = target ? resolveConversationRecall(params.cfg, params.options) : null;
  if (!target || !config) {
    return null;
  }
  return await runMemoryCorpusDeadline({
    operation: "memory_get",
    parentSignal: params.signal,
    run: async (signal) => {
      const missing = { path: params.relPath, text: "", corpus: "conversations" };
      try {
        const read = await readStoredConversation(
          { config, cfg: params.cfg, agentId: params.agentId, signal },
          { sessionId: target.sessionId, seq: target.seq, lines: params.lines },
        );
        return jsonResult(
          read
            ? { ...read, corpus: "conversations", status: "ok" }
            : { ...missing, status: "not_found" },
        );
      } catch (error) {
        if (signal.aborted) {
          throw error;
        }
        return jsonResult({ ...missing, status: "error", error: formatErrorMessage(error) });
      }
    },
  });
}
