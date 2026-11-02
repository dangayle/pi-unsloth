// SPDX-License-Identifier: MIT
// Pure mapping from Unsloth Studio model entries to Pi model definitions.
// Kept free of pi-ai imports so it can be unit-tested standalone.

import type { UnslothModelEntry, UnslothStatus } from "./client.ts";

export const PROVIDER_ID = "unsloth";

/**
 * Unsloth Studio marks non-chat models with a task. Chat models omit the
 * field, so only known media tasks are excluded and unknown ones pass
 * through.
 */
export const KNOWN_MEDIA_TASKS = new Set([
  "text-to-image",
  "image-to-image",
  "text-to-speech",
  "speech-to-text",
  "text-to-video",
  "image-to-video",
  "text-to-audio",
  "audio",
]);

/** Output cap mirroring unsloth_cli.opencode_output_limit for Pi. */
const MAX_OUTPUT_TOKENS = 32_000;

export function outputLimit(contextWindow: number): number {
  return Math.max(1, Math.min(Math.floor(contextWindow / 4), MAX_OUTPUT_TOKENS));
}

export function isChatEntry(entry: UnslothModelEntry): boolean {
  const task = entry.task;
  return task === undefined || !KNOWN_MEDIA_TASKS.has(task);
}

export function isLoadedEntry(entry: UnslothModelEntry): boolean {
  return entry.loaded || entry.status.value === "loaded";
}

export function contextWindowOf(entry: UnslothModelEntry): number {
  const window =
    entry.contextLength ?? entry.maxContextLength ?? entry.nativeContextLength;
  return window && window > 0 ? Math.floor(window) : 128_000;
}

export function nameOf(entry: UnslothModelEntry): string {
  return entry.displayName && entry.displayName !== entry.id ? entry.displayName : entry.id;
}

/** Whether the active-model status describes this entry. */
export function statusMatchesEntry(
  status: UnslothStatus | undefined,
  entry: UnslothModelEntry,
): boolean {
  if (!status) return false;
  const active = status.modelIdentifier ?? status.activeModel;
  if (!active) return false;
  return active === entry.id || active === entry.displayName || entry.id.endsWith(`/${active}`);
}

export interface PiChatModelDef {
  id: string;
  name: string;
  api: "openai-completions";
  provider: typeof PROVIDER_ID;
  baseUrl: string;
  reasoning: boolean;
  input: ("text" | "image")[];
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  contextWindow: number;
  maxTokens: number;
  compat: {
    supportsStore: boolean;
    supportsDeveloperRole: boolean;
    supportsReasoningEffort: boolean;
    supportsUsageInStreaming: boolean;
    supportsStrictMode: boolean;
    maxTokensField: "max_tokens";
    thinkingFormat?: "qwen-chat-template";
  };
  thinkingLevelMap?: Record<string, string | null>;
}

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

/**
 * Map one Unsloth model entry to a Pi chat model definition.
 *
 * `status` describes the active model; when it matches `entry`, runtime
 * capabilities (reasoning, vision) are trusted. Otherwise conservative
 * defaults apply, mirroring the built-in llama.cpp provider.
 */
export function toPiModel(
  entry: UnslothModelEntry,
  status: UnslothStatus | undefined,
  baseUrl: string,
): PiChatModelDef {
  const contextWindow = statusMatchesEntry(status, entry)
    ? (status?.contextLength ?? status?.maxContextLength ?? contextWindowOf(entry))
    : contextWindowOf(entry);
  const reasoning = statusMatchesEntry(status, entry) ? status?.supportsReasoning === true : false;
  const vision = statusMatchesEntry(status, entry) ? status?.isVision === true : false;
  return {
    id: entry.id,
    name: nameOf(entry),
    api: "openai-completions",
    provider: PROVIDER_ID,
    baseUrl,
    reasoning,
    input: vision ? ["text", "image"] : ["text"],
    cost: ZERO_COST,
    contextWindow: Math.floor(contextWindow),
    maxTokens: outputLimit(Math.floor(contextWindow)),
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsUsageInStreaming: true,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
      ...(reasoning ? { thinkingFormat: "qwen-chat-template" as const } : {}),
    },
    ...(reasoning
      ? {
          thinkingLevelMap: {
            off: "off",
            minimal: null,
            low: null,
            medium: "medium",
            high: null,
            xhigh: null,
          },
        }
      : {}),
  };
}

/** The catalog Pi exposes in /model: loaded, chat-capable models. */
export function toPiCatalog(
  entries: readonly UnslothModelEntry[],
  status: UnslothStatus | undefined,
  baseUrl: string,
): PiChatModelDef[] {
  return entries
    .filter((entry) => isLoadedEntry(entry) && isChatEntry(entry))
    .map((entry) => toPiModel(entry, status, baseUrl));
}
