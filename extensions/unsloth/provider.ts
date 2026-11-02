// SPDX-License-Identifier: MIT
// Pi provider glue: registers Unsloth Studio as a dynamic chat provider.
//
// Structure mirrors the built-in llama.cpp provider: a complete Provider from
// @earendil-works/pi-ai whose refreshModels pulls the live catalog from the
// Studio backend, plus delegated openai-completions streaming.

import { isModelType, type Provider } from "@earendil-works/pi-ai";
import { stream, streamSimple } from "@earendil-works/pi-ai/compat";
import {
  DEFAULT_STUDIO_URL,
  UnslothClient,
  normalizeStudioUrl,
  type UnslothModelEntry,
  type UnslothStatus,
} from "./client.ts";
import {
  PROVIDER_ID,
  toPiCatalog,
  type PiChatModelDef,
} from "./catalog.ts";
import {
  TokenCache,
  UNSLOTH_API_KEY_ENV,
  UNSLOTH_STUDIO_URL_ENV,
  findStudioPython,
  isMintedToken,
  mintToken,
  probeStudio,
  resolveAuth,
  resolveTarget,
  verifyAccess,
} from "./auth.ts";

const PROVIDER_NAME = "Unsloth";

export function inferenceUrl(serverUrl: string): string {
  return `${normalizeStudioUrl(serverUrl)}/v1`;
}

interface AuthCredentialShape {
  key?: string;
  env?: Record<string, string>;
}

function storedServerUrl(
  credential: AuthCredentialShape | undefined,
): string | undefined {
  const stored = credential?.env?.[UNSLOTH_STUDIO_URL_ENV];
  if (typeof stored === "string" && stored.trim()) return stored.trim();
  const fromEnv = process.env[UNSLOTH_STUDIO_URL_ENV];
  if (fromEnv && fromEnv.trim()) return fromEnv.trim();
  return undefined;
}

export interface UnslothProviderController {
  provider: Provider<"openai-completions">;
}

export function createUnslothProvider(): UnslothProviderController {
  let models: PiChatModelDef[] = [];
  // The cache is keyed by base URL, so a token minted for one server is never
  // handed to another when the configured URL changes.
  const tokenCache = new TokenCache(async () => {
    const minted = await mintToken({
      baseUrl: resolveTarget({ env: process.env }).baseUrl,
      env: process.env,
    });
    return isMintedToken(minted) ? minted : undefined;
  });

  const setCatalog = (catalog: readonly PiChatModelDef[]): void => {
    models = [...catalog];
  };

  const provider: Provider<"openai-completions"> = {
    id: PROVIDER_ID,
    name: PROVIDER_NAME,
    baseUrl: inferenceUrl(DEFAULT_STUDIO_URL),
    auth: {
      apiKey: {
        name: "Unsloth Studio server",
        login: async (interaction) => {
          const enteredUrl = await interaction.prompt({
            type: "text",
            message: "Unsloth Studio server URL",
            placeholder:
              process.env[UNSLOTH_STUDIO_URL_ENV] ?? DEFAULT_STUDIO_URL,
          });
          const serverUrl = normalizeStudioUrl(
            enteredUrl.trim() || process.env[UNSLOTH_STUDIO_URL_ENV] || DEFAULT_STUDIO_URL,
          );
          const enteredKey = (
            await interaction.prompt({
              type: "secret",
              message:
                "API key (optional — leave empty to use the same-user desktop token)",
            })
          ).trim();
          const apiKey = enteredKey || undefined;
          // Resolve a working credential for this exact URL before storing it.
          const resolved = await resolveAuth({
            serverUrl,
            apiKey,
            env: process.env,
            mint: async (baseUrl) => {
              const minted = await tokenCache.get(baseUrl);
              return minted;
            },
          });
          if (!resolved) {
            const probe = await probeStudio(serverUrl, { signal: interaction.signal });
            throw new Error(
              `Could not authenticate to the Unsloth server: ${probe.detail ?? "no usable credential"}. ` +
                "Check that Unsloth Studio is running at that URL, or provide an API key.",
            );
          }
          // Prove the credential is accepted by that exact server before
          // persisting it, so a wrong port cannot survive into a session.
          await verifyAccess({
            baseUrl: resolved.auth.baseUrl,
            apiKey: resolved.auth.apiKey,
            signal: interaction.signal,
          });
          return {
            type: "api_key" as const,
            key: resolved.auth.apiKey,
            env: { [UNSLOTH_STUDIO_URL_ENV]: resolved.auth.baseUrl },
          };
        },
        check: async ({ ctx, credential }) => {
          const serverUrl = storedServerUrl(credential) ?? (await ctx.env(UNSLOTH_STUDIO_URL_ENV));
          const apiKey = credential?.key ?? (await ctx.env(UNSLOTH_API_KEY_ENV));
          if (apiKey && apiKey.trim()) {
            return {
              type: "api_key" as const,
              source: credential ? "stored credential" : UNSLOTH_API_KEY_ENV,
            };
          }
          if (serverUrl || findStudioPython(process.env)) {
            return { type: "api_key" as const, source: "Unsloth desktop token" };
          }
          return undefined;
        },
        resolve: async ({ ctx, credential }) => {
          const serverUrl = storedServerUrl(credential) ?? (await ctx.env(UNSLOTH_STUDIO_URL_ENV));
          const resolved = await resolveAuth({
            serverUrl,
            apiKey: credential?.key,
            env: process.env,
            mint: (baseUrl) => tokenCache.get(baseUrl),
          });
          if (!resolved) return undefined;
          // The base comes from our deterministic target, never from a helper.
          return {
            auth: { apiKey: resolved.auth.apiKey, baseUrl: inferenceUrl(resolved.auth.baseUrl) },
            env: { ...credential?.env, [UNSLOTH_STUDIO_URL_ENV]: resolved.auth.baseUrl },
            source: resolved.source === "minted" ? "Unsloth desktop token" : resolved.source,
          };
        },
      },
    },
    getModels: () => models,
    refreshModels: async (context) => {
      if (context.stored) {
        const stored = context.stored.models.filter(
          (model) =>
            model.provider === PROVIDER_ID &&
            isModelType(model, "chat") &&
            model.api === "openai-completions",
        ) as PiChatModelDef[];
        if (!(await context.publish({ update: () => setCatalog(stored) }))) {
          return;
        }
      }
      if (!context.allowNetwork || context.signal.aborted) return;
      if (context.credential?.type !== "api_key") return;
      // The server we talk to is decided here, never by the server itself.
      const base = resolveTarget({
        serverUrl: storedServerUrl(context.credential),
        env: process.env,
      }).baseUrl;
      const client = new UnslothClient({
        baseUrl: base,
        apiKey: context.credential.key,
      });
      let entries: UnslothModelEntry[];
      let status: UnslothStatus | undefined;
      try {
        entries = await client.listModels({ signal: context.signal });
        status = await client.status({ signal: context.signal });
      } catch (error) {
        if ((error as { name?: string })?.name === "AbortError") return;
        // Keep the stored catalog when the server is unreachable.
        if (!models.length) throw error;
        return;
      }
      if (context.signal.aborted) return;
      const refreshed = toPiCatalog(entries, status, inferenceUrl(base));
      await context.publish({
        persist: { models: refreshed, checkedAt: Date.now() },
        update: () => setCatalog(refreshed),
      });
    },
    stream: (model, context, options) =>
      stream(model, context, options as Parameters<typeof stream>[2]),
    streamSimple: (model, context, options) => streamSimple(model, context, options),
  };

  return { provider };
}
