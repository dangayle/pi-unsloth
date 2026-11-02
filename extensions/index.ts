// SPDX-License-Identifier: MIT
// pi-unsloth: drive a running Unsloth Studio server from inside Pi.
//
// - Registers the `unsloth` chat provider whose catalog tracks the models
//   currently loaded on the Studio server, so they appear in /model.
// - Adds the /unsloth command to list every locally available model, load or
//   unload models, and inspect server status.

import type {
  ExtensionAPI,
  ExtensionCommandContext,
} from "@earendil-works/pi-coding-agent";
import {
  DEFAULT_STUDIO_URL,
  UnslothClient,
  connectionErrorMessage,
  normalizeStudioUrl,
  type ModelListEntry,
} from "./unsloth/client.ts";
import { findEntry, flagValue, numberFlag, tokenize } from "./unsloth/args.ts";
import { PROVIDER_ID } from "./unsloth/catalog.ts";
import { createUnslothProvider } from "./unsloth/provider.ts";
import {
  UNSLOTH_STUDIO_URL_ENV,
  probeStudio,
  resolveTarget,
} from "./unsloth/auth.ts";

const MAX_OPTION_LENGTH = 96;

export default function (pi: ExtensionAPI) {
  const { provider } = createUnslothProvider();
  pi.registerProvider(provider);

  pi.registerCommand("unsloth", {
    description:
      "Manage local Unsloth models: list every model on the server, load or unload models, and refresh the model catalog",
    handler: async (args, ctx) => unslothCommand(args, ctx),
  });
}

// ---------------------------------------------------------------------------
// Command implementation
// ---------------------------------------------------------------------------

async function unslothCommand(args: string, ctx: ExtensionCommandContext): Promise<void> {
  const tokens = tokenize(args);
  const subcommand = tokens[0]?.toLowerCase() ?? "";
  const rest = tokens.slice(1);

  if (subcommand === "help" || subcommand === "h" || subcommand === "--help") {
    printUsage(ctx);
    return;
  }

  const client = await configuredClient(ctx);
  if (!client) return;

  try {
    switch (subcommand) {
      case "load": {
        const model = rest.find((token) => !token.startsWith("--"));
        if (!model) {
          notify(ctx, "Usage: /unsloth load <model> [--variant V] [--context N]", "warning");
          return;
        }
        const variant = flagValue(rest, "--variant");
        const context = numberFlag(rest, "--context");
        await loadModelByName(ctx, client, model, { variant, maxSeqLength: context });
        return;
      }
      case "unload": {
        const model = rest.find((token) => !token.startsWith("--"));
        if (!model) {
          notify(ctx, "Usage: /unsloth unload <model>", "warning");
          return;
        }
        const force = rest.includes("--force");
        await unloadModelByName(ctx, client, model, force);
        return;
      }
      case "refresh": {
        await refreshCatalog(ctx);
        notify(ctx, "Unsloth catalog refreshed", "info");
        return;
      }
      case "status":
      case "st":
        await printStatus(ctx, client);
        return;
      case "list":
      case "ls":
        await printList(ctx, client);
        return;
      case "login":
        notify(ctx, "Run /login unsloth to configure the Unsloth server.", "info");
        return;
      case "":
        if (ctx.mode === "tui") {
          await interactivePicker(ctx, client);
        } else {
          await printList(ctx, client);
        }
        return;
      default:
        notify(ctx, `Unknown /unsloth subcommand: ${subcommand}`, "warning");
        printUsage(ctx);
    }
  } catch (error) {
    if ((error as { name?: string })?.name === "AbortError") {
      notify(ctx, "Unsloth operation cancelled.", "warning");
      return;
    }
    const message = connectionErrorMessage(error);
    notify(ctx, message, "error");
    print(ctx, message);
    if (message.includes("Could not connect")) {
      await diagnoseUnreachable((detail) => {
        notify(ctx, detail, "error");
        print(ctx, detail);
      });
    }
  }
}

// ---------------------------------------------------------------------------
// Server access
// ---------------------------------------------------------------------------

/** Strip a trailing /v1 so an inference base URL becomes the Studio root. */
function studioRoot(baseUrl: string | undefined): string {
  const base = normalizeStudioUrl(baseUrl ?? "");
  return base.endsWith("/v1") ? base.slice(0, -3) : base;
}

async function configuredClient(
  ctx: ExtensionCommandContext,
): Promise<UnslothClient | undefined> {
  const result = await ctx.modelRegistry.getProviderAuth(PROVIDER_ID);
  if (!result) {
    notify(
      ctx,
      `Unsloth is not configured. Run /login ${PROVIDER_ID}, or set ${UNSLOTH_STUDIO_URL_ENV}/UNSLOTH_API_KEY.`,
      "warning",
    );
    return undefined;
  }
  // The resolved auth already carries the base URL we deliberately chose.
  const baseUrl = studioRoot(result.auth.baseUrl);
  return new UnslothClient({ baseUrl, apiKey: result.auth.apiKey });
}

/** Name the actual problem when the configured server cannot be reached. */
async function diagnoseUnreachable(
  logger: (message: string) => void,
): Promise<void> {
  const target = resolveTarget({
    serverUrl: process.env[UNSLOTH_STUDIO_URL_ENV],
    env: process.env,
  });
  const probe = await probeStudio(target.baseUrl);
  const hint = target.explicit
    ? `the target came from ${UNSLOTH_STUDIO_URL_ENV}`
    : "that is the default; set UNSLOTH_STUDIO_URL if Unsloth runs elsewhere";
  logger(`${probe.detail ?? `could not reach ${target.baseUrl}`} — ${hint}.`);
}

async function refreshCatalog(ctx: ExtensionCommandContext): Promise<void> {
  const result = await ctx.modelRegistry.refresh({
    providers: [PROVIDER_ID],
    // The command already talked to the server, so keep this refresh live
    // even when Pi runs offline.
    allowNetwork: true,
    signal: AbortSignal.timeout(20_000),
  });
  if (result.aborted) throw new Error("Model catalog refresh timed out.");
  const refreshError = result.errors.get(PROVIDER_ID);
  if (refreshError) throw refreshError;
}

async function listEntries(client: UnslothClient): Promise<ModelListEntry[]> {
  return client.listModels({ signal: AbortSignal.timeout(20_000) });
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

function notify(
  ctx: ExtensionCommandContext,
  message: string,
  type: "info" | "warning" | "error" = "info",
): void {
  ctx.ui.notify(message, type);
}

function print(ctx: ExtensionCommandContext, message: string): void {
  if (ctx.mode !== "tui") console.log(message);
}

function formatContext(value: number | undefined): string {
  return value ? `${Math.floor(value).toLocaleString("en-US")} ctx` : "";
}

function formatBytes(value: number | undefined): string {
  if (!value) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 100 ? Math.round(size) : size.toFixed(1)} ${units[unit]}`;
}

function statusGlyph(entry: ModelListEntry): string {
  switch (entry.status.value) {
    case "loaded":
      return "●";
    case "loading":
    case "unloading":
      return "◐";
    default:
      return "○";
  }
}

function entryLabel(entry: ModelListEntry, glyph = true): string {
  const parts = [
    glyph ? statusGlyph(entry) : "",
    entry.id,
    entry.quant ? `[${entry.quant}]` : "",
    formatContext(entry.contextLength),
    entry.loaded ? "(loaded)" : "",
  ].filter(Boolean);
  const label = parts.join(" ");
  return label.length > MAX_OPTION_LENGTH
    ? `${label.slice(0, MAX_OPTION_LENGTH - 1)}…`
    : label;
}

function printUsage(ctx: ExtensionCommandContext): void {
  const usage = [
    "/unsloth                     interactive model picker (TUI)",
    "/unsloth list                list every locally available model",
    "/unsloth status              server and active-model status",
    "/unsloth load <model>        load a model [--variant Q8_0] [--context N]",
    "/unsloth unload <model>      unload a model [--force]",
    "/unsloth refresh             refresh the model catalog",
  ].join("\n");
  print(ctx, usage);
  notify(ctx, "Usage: /unsloth [list|status|load|unload|refresh]", "info");
}

async function printList(
  ctx: ExtensionCommandContext,
  client: UnslothClient,
): Promise<void> {
  const entries = await listEntries(client);
  if (entries.length === 0) {
    print(ctx, "No models available on the Unsloth server.");
    notify(ctx, "No models available on the Unsloth server.", "info");
    return;
  }
  const lines = [
    `${entries.length} model${entries.length === 1 ? "" : "s"} on ${client.baseUrl}`,
    ...entries.map((entry) => entryLabel(entry, false)),
  ];
  print(ctx, lines.join("\n"));
  notify(
    ctx,
    `${entries.filter((entry) => entry.loaded).length} of ${entries.length} models loaded`,
    "info",
  );
}

async function printStatus(
  ctx: ExtensionCommandContext,
  client: UnslothClient,
): Promise<void> {
  const entries = await listEntries(client);
  const status = await client.status({ signal: AbortSignal.timeout(20_000) });
  const loaded = entries.filter((entry) => entry.loaded);
  const lines = [
    `Server: ${client.baseUrl}`,
    `Active model: ${status.modelIdentifier ?? "(none)"}`,
    ...(status.ggufVariant ? [`GGUF variant: ${status.ggufVariant}`] : []),
    ...(status.contextLength
      ? [`Context: ${status.contextLength.toLocaleString("en-US")} tokens`]
      : []),
    ...(status.supportsReasoning ? ["Reasoning: supported"] : []),
    ...(status.isVision ? ["Vision: supported"] : []),
    `Loaded: ${loaded.length ? loaded.map((entry) => entry.id).join(", ") : "(none)"}`,
  ];
  print(ctx, lines.join("\n"));
  notify(ctx, lines.join(" · "), "info");
}

// ---------------------------------------------------------------------------
// Load / unload
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Load / unload
// ---------------------------------------------------------------------------

async function loadModelByName(
  ctx: ExtensionCommandContext,
  client: UnslothClient,
  name: string,
  options: { variant?: string; maxSeqLength?: number },
): Promise<void> {
  const entries = await listEntries(client);
  const target = findEntry(entries, name);
  if (!target) {
    notify(ctx, `No locally available model matches "${name}". Try /unsloth list.`, "error");
    return;
  }
  if (target.loaded) {
    notify(ctx, `${target.id} is already loaded.`, "info");
    return;
  }
  await loadModel(ctx, client, entries, target, options);
}

async function loadModel(
  ctx: ExtensionCommandContext,
  client: UnslothClient,
  entries: readonly ModelListEntry[],
  target: ModelListEntry,
  options: { variant?: string; maxSeqLength?: number },
): Promise<void> {
  const othersLoaded = entries.filter((entry) => entry.loaded && entry.id !== target.id);
  let replace = true;
  if (othersLoaded.length > 0) {
    if (ctx.mode === "tui") {
      const choice = await ctx.ui.select(
        `${othersLoaded.length} model${othersLoaded.length === 1 ? " is" : "s are"} loaded`,
        ["Unload them and load", "Keep them loaded and load", "Cancel"],
      );
      if (!choice || choice === "Cancel") return;
      replace = choice === "Unload them and load";
    }
    // In non-interactive mode the default is replace.
  }

  let variant = options.variant;
  if (!variant && target.id.toUpperCase().includes("GGUF") && ctx.mode === "tui") {
    try {
      const picked = await pickVariant(ctx, client, target);
      if (picked === CANCELLED) return;
      variant = picked;
    } catch {
      // Variant listing is a convenience; the server picks a default without it.
    }
  }

  if (replace) {
    for (const other of othersLoaded) {
      notify(ctx, `Unloading ${other.id}…`, "info");
      await client.unload({ modelPath: other.id });
    }
  }
  notify(ctx, `Loading ${target.id}${variant ? ` (${variant})` : ""}…`, "info");
  await client.load({
    modelPath: target.id,
    ggufVariant: variant,
    maxSeqLength: options.maxSeqLength,
  });
  await refreshCatalog(ctx);
  notify(ctx, `Loaded ${target.id} — select it in /model`, "info");
}

async function unloadModelByName(
  ctx: ExtensionCommandContext,
  client: UnslothClient,
  name: string,
  force: boolean,
): Promise<void> {
  const entries = await listEntries(client);
  const target = findEntry(entries, name);
  if (!target) {
    notify(ctx, `No locally available model matches "${name}".`, "error");
    return;
  }
  if (!target.loaded) {
    notify(ctx, `${target.id} is not loaded.`, "info");
    return;
  }
  if (ctx.mode === "tui" && !force) {
    const confirmed = await ctx.ui.confirm("Unload model?", target.id);
    if (!confirmed) return;
  }
  notify(ctx, `Unloading ${target.id}…`, "info");
  await client.unload({ modelPath: target.id });
  await refreshCatalog(ctx);
  notify(ctx, `Unloaded ${target.id}`, "info");
}

const CANCELLED = Symbol("cancelled");

/** Pick a GGUF quant variant; undefined means the server's default. */
async function pickVariant(
  ctx: ExtensionCommandContext,
  client: UnslothClient,
  target: ModelListEntry,
): Promise<string | undefined | typeof CANCELLED> {
  const variants = await client.ggufVariants(target.id, {
    signal: AbortSignal.timeout(20_000),
  });
  if (variants.length === 0) return undefined;
  const options = [
    "Default (server picks)",
    ...variants.map((variant) => {
      const details = [
        variant.quant,
        formatBytes(variant.sizeBytes),
        variant.downloaded ? "cached" : "",
      ]
        .filter(Boolean)
        .join(" · ");
      return details;
    }),
  ];
  const choice = await ctx.ui.select(`GGUF variant for ${target.id}`, options);
  if (choice === undefined) return CANCELLED;
  if (choice === options[0]) return undefined;
  const index = options.indexOf(choice);
  return variants[index - 1]?.quant;
}

// ---------------------------------------------------------------------------
// Interactive picker
// ---------------------------------------------------------------------------

async function interactivePicker(
  ctx: ExtensionCommandContext,
  client: UnslothClient,
): Promise<void> {
  while (true) {
    let entries: ModelListEntry[];
    try {
      entries = await listEntries(client);
    } catch (error) {
      notify(ctx, connectionErrorMessage(error), "error");
      return;
    }
    const options = ["Refresh list", "Done"];
    const byLabel = new Map<string, ModelListEntry>();
    for (const entry of entries) {
      const label = entryLabel(entry);
      byLabel.set(label, entry);
      options.push(label);
    }
    const choice = await ctx.ui.select(
      `Unsloth models — ${client.baseUrl}`,
      options,
    );
    if (!choice || choice === "Done") return;
    if (choice === "Refresh list") continue;
    const target = byLabel.get(choice);
    if (!target) continue;
    const action = await ctx.ui.select(
      target.id,
      target.loaded
        ? ["Unload", "Back"]
        : target.status.value === "loading"
          ? ["Back"]
          : ["Load", "Back"],
    );
    if (!action || action === "Back") continue;
    try {
      if (action === "Unload") {
        await unloadModelByName(ctx, client, target.id, false);
      } else {
        await loadModel(ctx, client, entries, target, {});
      }
    } catch (error) {
      if ((error as { name?: string })?.name === "AbortError") {
        notify(ctx, "Unsloth operation cancelled.", "warning");
      } else {
        notify(ctx, connectionErrorMessage(error), "error");
      }
    }
  }
}
