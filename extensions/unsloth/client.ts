// SPDX-License-Identifier: MIT
// REST client for a running Unsloth Studio backend.
//
// Reuses Unsloth Studio's HTTP contract rather than reimplementing model
// management: the padded load/unload responses, the `_deferred_error`
// in-band failure channel, and the OpenAI-compatible /v1 endpoints all
// belong to the Studio backend (see unsloth_cli/_inference.py).

export const DEFAULT_STUDIO_URL = "http://127.0.0.1:8888";
const USER_AGENT = "pi-unsloth";
// The Studio backend pads load/unload bodies and commits 200 early, so a
// late failure rides in-band under this key; a bare 200 must read as success.
const DEFERRED_ERROR_KEY = "_deferred_error";

export type UnslothModelStatusValue =
  | "loaded"
  | "unloaded"
  | "loading"
  | "unloading"
  | "failed"
  | string;

export interface UnslothModelEntry {
  id: string;
  loaded: boolean;
  status: { value?: UnslothModelStatusValue };
  quant?: string;
  contextLength?: number;
  maxContextLength?: number;
  nativeContextLength?: number;
  vramFitContextLength?: number;
  displayName?: string;
  /** Media models carry a task like "text-to-image"; chat models omit it. */
  task?: string;
  ownedBy?: string;
  created?: number;
}

export interface UnslothStatus {
  modelIdentifier?: string;
  activeModel?: string;
  isGguf?: boolean;
  ggufVariant?: string;
  contextLength?: number;
  maxContextLength?: number;
  supportsReasoning?: boolean;
  reasoningStyle?: string;
  supportsTools?: boolean;
  isVision?: boolean;
}

export interface GgufVariant {
  filename: string;
  quant: string;
  contextLength?: number;
  sizeBytes?: number;
  downloaded?: boolean;
}

export interface LoadOptions {
  modelPath: string;
  ggufVariant?: string;
  maxSeqLength?: number;
  signal?: AbortSignal;
}

export interface UnloadOptions {
  modelPath: string;
  signal?: AbortSignal;
}

export interface ClientOptions {
  baseUrl: string;
  apiKey?: string;
  fetchFn?: typeof fetch;
}

export class UnslothError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "UnslothError";
    this.status = status;
  }
}

export function isConnectionError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const message = `${error.name} ${error.message}`.toLowerCase();
  return (
    message.includes("fetch failed") ||
    message.includes("network") ||
    message.includes("timeout") ||
    message.includes("terminated") ||
    message.includes("connect")
  );
}

export function connectionErrorMessage(error: unknown): string {
  if (isConnectionError(error)) {
    return "Could not connect to the Unsloth server.";
  }
  return error instanceof Error ? error.message : String(error);
}

export function normalizeStudioUrl(base: string): string {
  return base.trim().replace(/\/+$/, "");
}

/** A locally available (or loaded) model known to the Studio backend. */
export interface ModelListEntry extends UnslothModelEntry {}

function parseModelEntry(raw: unknown): ModelListEntry | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const entry = raw as Record<string, unknown>;
  const id = typeof entry.id === "string" ? entry.id : undefined;
  if (!id) return undefined;
  const statusValue = (entry.status as { value?: unknown } | undefined)?.value;
  return {
    id,
    loaded: entry.loaded === true,
    status: { value: typeof statusValue === "string" ? statusValue : undefined },
    quant: typeof entry.quant === "string" ? entry.quant : undefined,
    contextLength: toInt(entry.context_length),
    maxContextLength: toInt(entry.max_context_length),
    nativeContextLength: toInt(entry.native_context_length),
    vramFitContextLength: toInt(entry.vram_fit_context_length),
    displayName: typeof entry.display_name === "string" ? entry.display_name : undefined,
    task: typeof entry.task === "string" ? entry.task : undefined,
    ownedBy: typeof entry.owned_by === "string" ? entry.owned_by : undefined,
    created: toInt(entry.created),
  };
}

function toInt(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export class UnslothClient {
  readonly baseUrl: string;
  readonly apiKey: string | undefined;
  private readonly fetchFn: typeof fetch;

  constructor(options: ClientOptions) {
    this.baseUrl = normalizeStudioUrl(options.baseUrl);
    this.apiKey = options.apiKey;
    this.fetchFn = options.fetchFn ?? fetch;
  }

  private async request(
    method: string,
    path: string,
    body?: unknown,
    signal?: AbortSignal,
  ): Promise<unknown> {
    const headers: Record<string, string> = {
      "User-Agent": USER_AGENT,
    };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const response = await this.fetchFn(`${this.baseUrl}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
    const text = await response.text();
    if (!response.ok) {
      throw new UnslothError(parseErrorDetail(text, response.status), response.status);
    }
    return parseJson(text);
  }

  /** Padded load/unload responses must complete with a non-empty JSON body. */
  private async requestPadded(
    method: string,
    path: string,
    body: unknown,
    signal?: AbortSignal,
  ): Promise<Record<string, unknown>> {
    const headers: Record<string, string> = {
      "User-Agent": USER_AGENT,
      "Content-Type": "application/json",
    };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    const response = await this.fetchFn(`${this.baseUrl}${path}`, {
      method,
      headers,
      body: JSON.stringify(body),
      signal,
    });
    const text = await response.text();
    if (!response.ok) {
      throw new UnslothError(parseErrorDetail(text, response.status), response.status);
    }
    const parsed = parseJson(text);
    // Mirror unsloth_cli.require_completed_padded_body: an empty or truncated
    // body means the connection closed before the server's reply arrived.
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed) || Object.keys(parsed).length === 0) {
      throw new UnslothError(
        `${this.baseUrl}${path} did not report completion: the connection closed before the server's reply arrived. Check the model's status before retrying.`,
      );
    }
    const object = parsed as Record<string, unknown>;
    const deferred = object[DEFERRED_ERROR_KEY];
    if (deferred && typeof deferred === "object") {
      const detail = (deferred as Record<string, unknown>).detail;
      const status = toInt((deferred as Record<string, unknown>).status_code) ?? 500;
      throw new UnslothError(
        typeof detail === "string" && detail ? detail : JSON.stringify(detail),
        status,
      );
    }
    return object;
  }

  /** GET /api/inference/models — every locally available model plus the loaded flag. */
  async listModels(options?: { signal?: AbortSignal }): Promise<ModelListEntry[]> {
    const data = await this.request("GET", "/api/inference/models", undefined, options?.signal);
    const entries = (data as { data?: unknown } | undefined)?.data;
    if (!Array.isArray(entries)) return [];
    return entries.map(parseModelEntry).filter((entry): entry is ModelListEntry => entry !== undefined);
  }

  /** GET /api/inference/status — the active model's runtime status. */
  async status(options?: { signal?: AbortSignal }): Promise<UnslothStatus> {
    const raw = (await this.request(
      "GET",
      "/api/inference/status",
      undefined,
      options?.signal,
    )) as Record<string, unknown>;
    const status: UnslothStatus = {};
    status.modelIdentifier = str(raw.model_identifier);
    status.activeModel = str(raw.active_model);
    status.isGguf = raw.is_gguf === true;
    status.ggufVariant = str(raw.gguf_variant);
    status.contextLength = toInt(raw.context_length);
    status.maxContextLength = toInt(raw.max_context_length);
    status.supportsReasoning = raw.supports_reasoning === true;
    status.reasoningStyle = str(raw.reasoning_style);
    status.supportsTools = raw.supports_tools === true;
    status.isVision = raw.is_vision === true;
    return status;
  }

  /** GET /api/models/gguf-variants?repo_id=... — quant variants for a GGUF repo. */
  async ggufVariants(
    repoId: string,
    options?: { signal?: AbortSignal },
  ): Promise<GgufVariant[]> {
    const query = new URLSearchParams({ repo_id: repoId });
    const raw = (await this.request(
      "GET",
      `/api/models/gguf-variants?${query}`,
      undefined,
      options?.signal,
    )) as { variants?: unknown } | undefined;
    const variants = raw?.variants;
    if (!Array.isArray(variants)) return [];
    const result: GgufVariant[] = [];
    for (const item of variants) {
      if (!item || typeof item !== "object") continue;
      const v = item as Record<string, unknown>;
      const quant = typeof v.quant === "string" ? v.quant : undefined;
      if (!quant) continue;
      result.push({
        quant,
        filename: typeof v.filename === "string" ? v.filename : "",
        contextLength: toInt(v.context_length),
        sizeBytes: toInt(v.size_bytes),
        downloaded: v.downloaded === true,
      });
    }
    return result;
  }

  /** POST /api/inference/load — load a model into memory on the Studio server. */
  async load(options: LoadOptions): Promise<Record<string, unknown>> {
    const payload: Record<string, unknown> = {
      model_path: options.modelPath,
    };
    if (options.ggufVariant) payload.gguf_variant = options.ggufVariant;
    if (options.maxSeqLength && options.maxSeqLength > 0) {
      payload.max_seq_length = options.maxSeqLength;
    }
    return this.requestPadded("POST", "/api/inference/load", payload, options.signal);
  }

  /** POST /api/inference/unload — remove a model from memory. */
  async unload(options: UnloadOptions): Promise<Record<string, unknown>> {
    return this.requestPadded("POST", "/api/inference/unload", {
      model_path: options.modelPath,
    }, options.signal);
  }
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

function parseErrorDetail(text: string, status: number): string {
  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    const detail = parsed.detail;
    if (typeof detail === "string" && detail) return detail;
    if (detail && typeof detail === "object") {
      const message = (detail as Record<string, unknown>).message;
      if (typeof message === "string" && message) return message;
    }
    const error = parsed.error;
    if (error && typeof error === "object") {
      const message = (error as Record<string, unknown>).message;
      if (typeof message === "string" && message) return message;
    }
  } catch {
    // Fall through to the status-based message.
  }
  return `Unsloth server error (HTTP ${status})`;
}

function parseJson(text: string): unknown {
  if (!text.trim()) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}
