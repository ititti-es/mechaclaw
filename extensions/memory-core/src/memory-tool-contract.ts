import { resolveSessionAgentIdStrict } from "openclaw/plugin-sdk/agent-scope-runtime";
import {
  asToolParamsRecord,
  type AnyAgentTool,
  resolveMemorySearchIndexConfig,
  type MemoryPromptSectionBuilder,
  type OpenClawConfig,
} from "openclaw/plugin-sdk/memory-core-host-runtime-core";
import type { OpenClawPluginToolContext } from "openclaw/plugin-sdk/plugin-entry";
import type { TSchema } from "typebox";
import { resolveConversationRecallForSession } from "./conversation-recall.js";
import type { MemoryCoreAcquireLocalService } from "./memory/embedding-local-service.js";

export type MemoryToolOptions = {
  config?: OpenClawConfig;
  getConfig?: () => OpenClawConfig | undefined;
  agentId?: string;
  agentSessionKey?: string;
  sandboxed?: boolean;
  oneShotCliRun?: boolean;
  conversationRecall?: OpenClawPluginToolContext["conversationRecall"];
  activeProjectKeys?: readonly string[];
  /** Runtime owner bit for the turn; conversation recall is only offered to the owner. */
  senderIsOwner?: boolean;
  acquireLocalService?: MemoryCoreAcquireLocalService;
};

const MemorySearchSchema = {
  type: "object",
  properties: {
    query: { type: "string" },
    maxResults: { type: "integer", minimum: 1 },
    minScore: { type: "number" },
    corpus: { type: "string", enum: ["memory", "wiki", "all", "sessions"] },
  },
  required: ["query"],
  additionalProperties: false,
} as const satisfies TSchema;

const MemoryGetSchema = {
  type: "object",
  properties: {
    path: { type: "string" },
    from: { type: "integer", minimum: 1 },
    lines: { type: "integer", minimum: 1 },
    corpus: { type: "string", enum: ["memory", "wiki", "all"] },
  },
  required: ["path"],
  additionalProperties: false,
} as const satisfies TSchema;

type MemorySourceContract = Readonly<{ files: string; search: string; conversations: boolean }>;

function resolveMemorySourceContract(
  settings: NonNullable<ReturnType<typeof resolveMemorySearchIndexConfig>>,
  conversations: boolean,
): MemorySourceContract {
  const files = [
    "MEMORY.md, USER.md, Markdown files recursively under memory/",
    settings.extraPaths.length > 0 ? "configured extra paths" : "",
  ]
    .filter(Boolean)
    .join(", ");
  const search = settings.searchSources.includes("sessions")
    ? `${files}, indexed session transcripts`
    : files;
  return {
    files,
    search: conversations
      ? `${search}, and every past conversation with the user across OpenClaw, opencode and Claude Code (reranked)`
      : search,
    conversations,
  };
}

export function resolveMemoryToolContext(options: MemoryToolOptions) {
  const cfg = options.getConfig ? options.getConfig() : options.config;
  if (!cfg) {
    return null;
  }
  const agentId = resolveSessionAgentIdStrict({
    sessionKey: options.agentSessionKey,
    config: cfg,
    agentId: options.agentId,
  });
  // Tool schemas and guidance need source policy; provider validation belongs to execution.
  const settings = resolveMemorySearchIndexConfig(cfg, agentId);
  return settings
    ? {
        cfg,
        agentId,
        settings,
        sources: resolveMemorySourceContract(
          settings,
          resolveConversationRecallForSession(cfg, {
            sessionKey: options.agentSessionKey,
            sandboxed: options.sandboxed,
            scopedRecall: options.conversationRecall !== undefined,
            senderIsOwner: options.senderIsOwner,
          }) !== null,
        ),
      }
    : null;
}

const SEARCH_CORPUS_OUTCOME_GUIDANCE =
  "Corpus outcomes cover each requested corpus; a corpus warning means results are partial and must be surfaced to the user.";
const GET_READ_OUTCOME_GUIDANCE =
  "status=ok means the requested excerpt was read; status=not_found means every requested available corpus missed; status=error means the requested read failed, not that memory is disabled.";

const CONVERSATION_SEARCH_GUIDANCE = (enabled: boolean) =>
  enabled
    ? " Conversation hits (corpus=conversations) are not memory files: their path is `conversation:<session>#<seq>`, which memory_get reads. `corpus=sessions` searches conversations only; `corpus=memory` searches the notes only."
    : "";

const CONVERSATION_GET_GUIDANCE = (enabled: boolean) =>
  enabled
    ? " Also reads a conversation hit from memory_search by its `conversation:<session>#<seq>` path: returns the messages around that hit, and `lines` sets how many."
    : "";

export const MEMORY_SEARCH_TOOL_CONTRACT = {
  label: "Memory Search",
  name: "memory_search",
  parameters: MemorySearchSchema,
  prepareArguments: (args: unknown) => {
    const params = asToolParamsRecord(args);
    if (!Object.hasOwn(params, "min_score") && !Object.hasOwn(params, "max_results")) {
      return args;
    }
    const normalized = { ...params };
    // Explicit camelCase values win, including invalid values that validation must reject.
    for (const [alias, key] of [
      ["min_score", "minScore"],
      ["max_results", "maxResults"],
    ] as const) {
      if (Object.hasOwn(normalized, alias)) {
        if (!Object.hasOwn(normalized, key)) {
          normalized[key] = normalized[alias];
        }
        delete normalized[alias];
      }
    }
    return normalized;
  },
  describe: ({ search, conversations }: MemorySourceContract) =>
    `Mandatory recall step: semantically search ${search} before answering questions about prior work, decisions, dates, people, preferences, or todos. Session results are transcript search references, not readable memory-file paths.${CONVERSATION_SEARCH_GUIDANCE(conversations)} Optional \`corpus=wiki\` or \`corpus=all\` also searches registered compiled-wiki supplements. \`corpus=memory\` restricts hits to indexed memory files (excludes session transcript chunks from ranking). \`corpus=sessions\` searches indexed session transcripts under the same visibility rules as session history tools and returns unavailable when semantic session indexing is disabled. ${SEARCH_CORPUS_OUTCOME_GUIDANCE} If response has disabled=true or stale=true, tell the user and include the warning/action guidance.`,
} as const;

export const MEMORY_GET_TOOL_CONTRACT = {
  label: "Memory Get",
  name: "memory_get",
  parameters: MemoryGetSchema,
  describe: ({ files, conversations }: MemorySourceContract) =>
    `Safe exact excerpt read from ${files}. Session transcript paths are unsupported; use the available session-history workflow for session hits.${CONVERSATION_GET_GUIDANCE(conversations)} Defaults to a bounded excerpt when lines are omitted and includes truncation/continuation info when more content exists. \`corpus=wiki\` reads registered compiled-wiki supplements. ${GET_READ_OUTCOME_GUIDANCE} ${SEARCH_CORPUS_OUTCOME_GUIDANCE}`,
} as const;

export type MemoryToolContract = (
  | typeof MEMORY_SEARCH_TOOL_CONTRACT
  | typeof MEMORY_GET_TOOL_CONTRACT
) & { prepareArguments?: AnyAgentTool["prepareArguments"] };

export function buildMemoryPromptSection(
  { availableTools, citationsMode }: Parameters<MemoryPromptSectionBuilder>[0],
  options: { conversations?: boolean } = {},
): string[] {
  const hasMemorySearch = availableTools.has("memory_search");
  const hasMemoryGet = availableTools.has("memory_get");
  if (!hasMemorySearch && !hasMemoryGet) {
    return [];
  }

  // Code mode may defer tool descriptions; recall and disclosure policy must stay here.
  const guidance = hasMemorySearch
    ? `Before answering anything about prior work, decisions, dates, people, preferences, or todos: run memory_search${
        hasMemoryGet ? "; for memory-file hits, use memory_get to pull only the needed lines" : ""
      }. If low confidence after search, say you checked.`
    : "Before answering anything about prior work, decisions, dates, people, preferences, or todos that point to a specific memory file: run memory_get to pull only the needed lines. If low confidence after reading, say you checked.";
  const conversationGuidance = [
    "For the owner, memory_search also recalls past conversations from every harness (OpenClaw, opencode, Claude Code), ranked by relevance. For a hit whose path starts with conversation:, call memory_get with that exact path to read the messages around it. Prefer memory_search over sessions_search for recall.",
  ];
  const sessionGuidance = !hasMemorySearch
    ? []
    : options.conversations === true
      ? conversationGuidance
      : [
          availableTools.has("sessions_search")
            ? `For session hits, use sessions_search with distinctive snippet text (and sessionKey set to the transcript ID when known)${
                availableTools.has("sessions_history")
                  ? ", then sessions_history with the returned sessionKey, messageId, and sessionId for a bounded sanitized excerpt"
                  : "; exact session history is unavailable with the enabled tools"
              }.`
            : availableTools.has("sessions_history")
              ? "For session hits, use sessions_history with a known session key or transcript ID and a small limit; paginate its returned history metadata to locate the excerpt."
              : "Session hits are search snippets only; exact session history is unavailable with the enabled tools.",
          "Session search line numbers are not history offsets. Never read raw transcript files to expand session hits.",
        ];
  const outcomeGuidance =
    "Report partial, unavailable, or stale recall to the user, including returned warning and action guidance.";
  const citationGuidance =
    citationsMode === "off"
      ? "Citations are disabled: do not mention file paths or line numbers in replies unless the user explicitly asks."
      : "Citations: include Source: <path#line> when it helps the user verify memory snippets.";
  return ["## Memory Recall", guidance, ...sessionGuidance, outcomeGuidance, citationGuidance, ""];
}
