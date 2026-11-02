// SPDX-License-Identifier: MIT
// Authentication against a running Unsloth Studio backend.
//
// Port selection is deterministic: this module decides which base URL to talk
// to (stored credential, then UNSLOTH_STUDIO_URL, then the default) and never
// lets a helper choose a different one. Unsloth's own
// `unsloth_cli._inference` functions are reused for the two things that need
// local secrets — `verify_studio_identity` (HMAC proof that the process on
// *that exact address* holds the secret from studio.db) and `_studio_token`
// (self-issued desktop JWT) — plus `is_loopback_url` for the safety rule that
// a self-issued token is only ever sent to a loopback address.
//
// `find_studio_server()` is deliberately NOT used: it reports a server it
// discovered, preferring "the server this install recorded", so with more than
// one server running it can answer with a port nobody asked about.
//
// A user API key (UNSLOTH_API_KEY, or one stored through /login unsloth) takes
// precedence and works against any server URL, including non-loopback.

import { spawn, type ChildProcess, type SpawnOptions } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { DEFAULT_STUDIO_URL, normalizeStudioUrl } from "./client.ts";

export const UNSLOTH_STUDIO_URL_ENV = "UNSLOTH_STUDIO_URL";
export const UNSLOTH_API_KEY_ENV = "UNSLOTH_API_KEY";

/** Margin (seconds) before a minted JWT's expiry at which we re-mint. */
const TOKEN_REFRESH_MARGIN_SECONDS = 300;

/** Cheap authenticated endpoint: Unsloth returns loaded models without a disk scan. */
const ACCESS_CHECK_PATH = "/api/inference/loaded-models";

export type Env = Record<string, string | undefined>;

/** Test seam: a spawn-compatible function. */
export type SpawnFn = (
  command: string,
  args: readonly string[],
  options: SpawnOptions,
) => ChildProcess;

// ---------------------------------------------------------------------------
// Which server?
// ---------------------------------------------------------------------------

export interface AuthTarget {
  /** Absolute base URL, no trailing slash. Never chosen by a helper. */
  baseUrl: string;
  /** Where the URL came from, for error messages. */
  origin: "credential" | "env" | "default";
  /** True when the user pointed somewhere specific, so failures should be loud. */
  explicit: boolean;
}

export interface TargetInput {
  /** URL from a stored credential. */
  serverUrl?: string;
  env: Env;
}

/** Deterministic server selection: credential, then env, then the default. */
export function resolveTarget(input: TargetInput): AuthTarget {
  const fromCredential = input.serverUrl?.trim();
  if (fromCredential) {
    return { baseUrl: normalizeStudioUrl(fromCredential), origin: "credential", explicit: true };
  }
  const fromEnv = input.env[UNSLOTH_STUDIO_URL_ENV]?.trim();
  if (fromEnv) {
    return { baseUrl: normalizeStudioUrl(fromEnv), origin: "env", explicit: true };
  }
  return { baseUrl: normalizeStudioUrl(DEFAULT_STUDIO_URL), origin: "default", explicit: false };
}

// ---------------------------------------------------------------------------
// Locating the bundled venv
// ---------------------------------------------------------------------------

/** Where the bundled `unsloth_studio` venv lives on this machine. */
export function studioHome(env: Env): string {
  const configured = env.UNSLOTH_STUDIO_HOME ?? env.STUDIO_HOME ?? env.UNSLOTH_HOME;
  if (configured && configured.trim()) return configured.trim();
  return path.join(os.homedir(), ".unsloth", "studio");
}

/** The Python interpreter of the bundled unsloth_studio venv, if present. */
export function findStudioPython(env: Env): string | undefined {
  const explicit = env.UNSLOTH_PYTHON;
  if (explicit && explicit.trim()) return explicit.trim();
  const venv = path.join(studioHome(env), "unsloth_studio");
  const candidates =
    process.platform === "win32"
      ? [path.join(venv, "Scripts", "python.exe"), path.join(venv, "Scripts", "python")]
      : [path.join(venv, "bin", "python"), path.join(venv, "bin", "python3")];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return undefined;
}

/** site-packages dir of the bundled venv, used as the script's sys.path root. */
function sitePackagesOf(pythonPath: string): string {
  return path.join(path.dirname(pythonPath), "..", "Lib", "site-packages");
}

/** node:child_process requires string values; drop undefined entries. */
export function definedEnv(env: Env): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(env)) {
    if (typeof value === "string") result[key] = value;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Reaching the server
// ---------------------------------------------------------------------------

export interface ProbeOptions {
  fetchFn?: typeof fetch;
  signal?: AbortSignal;
  timeoutMs?: number;
}

export interface ProbeResult {
  reachable: boolean;
  isStudio: boolean;
  detail?: string;
}

/**
 * Check that something answers at `baseUrl` and that it is an Unsloth Studio
 * backend. Used to fail with a precise message instead of a generic error.
 */
export async function probeStudio(
  baseUrl: string,
  options: ProbeOptions = {},
): Promise<ProbeResult> {
  const base = normalizeStudioUrl(baseUrl);
  const fetchFn = options.fetchFn ?? fetch;
  const signal = options.signal ?? AbortSignal.timeout(options.timeoutMs ?? 5_000);
  let response: Response;
  try {
    response = await fetchFn(`${base}/api/health`, { signal });
  } catch (error) {
    return {
      reachable: false,
      isStudio: false,
      detail: `nothing is answering at ${base} (${error instanceof Error ? error.message : String(error)})`,
    };
  }
  if (!response.ok) {
    return {
      reachable: true,
      isStudio: false,
      detail: `${base} answered HTTP ${response.status} for /api/health`,
    };
  }
  const body = (await response.json().catch(() => undefined)) as
    | Record<string, unknown>
    | undefined;
  const isStudio =
    body?.service === "Unsloth UI Backend" || typeof body?.studio_root_id === "string";
  return {
    reachable: true,
    isStudio,
    detail: isStudio
      ? undefined
      : `${base} answered, but it is not an Unsloth Studio server`,
  };
}

export interface AccessCheckInput {
  baseUrl: string;
  apiKey?: string;
  fetchFn?: typeof fetch;
  signal?: AbortSignal;
  timeoutMs?: number;
}

/**
 * Prove that `apiKey` is accepted by the server at exactly `baseUrl`.
 *
 * This is the end-to-end guarantee behind "am I talking to the right port?":
 * a token minted for one Studio home is rejected by a server running another,
 * and catching that here names both the URL and the problem instead of failing
 * later in the middle of a model load.
 */
export async function verifyAccess(input: AccessCheckInput): Promise<void> {
  const base = normalizeStudioUrl(input.baseUrl);
  const fetchFn = input.fetchFn ?? fetch;
  const signal = input.signal ?? AbortSignal.timeout(input.timeoutMs ?? 10_000);
  const headers: Record<string, string> = { "User-Agent": "pi-unsloth" };
  if (input.apiKey) headers.Authorization = `Bearer ${input.apiKey}`;
  let response: Response;
  try {
    response = await fetchFn(`${base}${ACCESS_CHECK_PATH}`, { headers, signal });
  } catch (error) {
    throw new Error(
      `Could not reach the Unsloth server at ${base}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (response.status === 401 || response.status === 403) {
    throw new Error(
      `${base} rejected the credentials (HTTP ${response.status}). ` +
        `It may be an Unsloth install whose studio.db differs from this one; ` +
        `set ${UNSLOTH_API_KEY_ENV} to an API key issued by that server.`,
    );
  }
  if (!response.ok) {
    throw new Error(`${base} answered HTTP ${response.status} for ${ACCESS_CHECK_PATH}`);
  }
}

// ---------------------------------------------------------------------------
// Minting a desktop token for one specific base URL
// ---------------------------------------------------------------------------

export interface MintedToken {
  baseUrl: string;
  token: string;
  /** JWT `exp` claim, in seconds since the epoch. */
  expiresAt?: number;
}

export interface MintFailure {
  reason: string;
}

export type MintOutcome = MintedToken | MintFailure | undefined;
export type TokenMinter = () => Promise<MintedToken | undefined>;

export interface MintOptions {
  /** The exact server to obtain a token for. Embedded in the helper script. */
  baseUrl: string;
  pythonPath?: string;
  env: Env;
  spawnFn?: SpawnFn;
  signal?: AbortSignal;
  timeoutMs?: number;
}

export function isMintedToken(value: MintOutcome): value is MintedToken {
  return value !== undefined && "token" in value;
}

/**
 * Mint a desktop token for exactly `options.baseUrl`.
 *
 * Returns a `MintFailure` when the address is not loopback or the server there
 * cannot prove the local secret, and `undefined` when the helper cannot run at
 * all (no venv, spawn error, abort) — that distinction lets callers stay quiet
 * about "Unsloth isn't installed" but loud about "you pointed me at the wrong
 * server".
 */
export async function mintToken(options: MintOptions): Promise<MintOutcome> {
  const baseUrl = normalizeStudioUrl(options.baseUrl);
  const pythonPath = options.pythonPath ?? findStudioPython(options.env);
  if (!pythonPath) return undefined;
  const site = sitePackagesOf(pythonPath);
  const home = studioHome(options.env);
  const script = [
    "import json, sys",
    `sys.path.insert(0, ${JSON.stringify(site)})`,
    `sys.path.insert(0, ${JSON.stringify(home)})`,
    "from unsloth_cli._inference import verify_studio_identity, _studio_token, is_loopback_url",
    `base = ${JSON.stringify(baseUrl)}`,
    'out = {"base": base, "verified": False}',
    "if not is_loopback_url(base):",
    '    out["reason"] = "the address is not loopback, so a self-issued desktop token cannot be used"',
    "elif not verify_studio_identity(base):",
    '    out["reason"] = "the server at that address did not prove the shared secret from studio.db"',
    "else:",
    "    out[\"verified\"] = True",
    "    token = _studio_token()",
    "    if token:",
    '        out["token"] = token',
    "    else:",
    '        out["reason"] = "unsloth_cli could not self-issue a desktop token"',
    "print(json.dumps(out))",
  ].join("\n");
  const spawnFn = options.spawnFn ?? spawn;
  const child = spawnFn(pythonPath, ["-c", script], {
    cwd: site,
    // Forward the caller's environment explicitly: the helper reads Studio
    // locations from it (STUDIO_HOME and friends).
    env: definedEnv(options.env),
    windowsHide: true,
    signal: options.signal,
    timeout: options.timeoutMs,
  });
  let stdout = "";
  let stderr = "";
  child.stdout?.on("data", (chunk: Buffer) => {
    stdout += chunk.toString();
  });
  child.stderr?.on("data", (chunk: Buffer) => {
    stderr += chunk.toString();
  });
  const { exitCode, signalCode, spawnError } = await new Promise<{
    exitCode: number | null;
    signalCode: NodeJS.Signals | null;
    spawnError: boolean;
  }>((resolve) => {
    child.on("error", () => {
      // Spawn failures are a missing/broken install, not an auth failure.
      resolve({ exitCode: null, signalCode: null, spawnError: true });
    });
    child.on("close", (code: number | null, signal: NodeJS.Signals | null) =>
      resolve({ exitCode: code, signalCode: signal, spawnError: false }),
    );
  });
  if (options.signal?.aborted) return undefined;
  if (spawnError) return undefined;
  if (exitCode !== 0 || signalCode) {
    if (signalCode) throw new Error("Unsloth token minting timed out.");
    const detail = stderr.trim().split("\n").at(-1) ?? "";
    throw new Error(`Unsloth auth helper failed: ${detail || `exit code ${exitCode}`}`);
  }
  const lastLine = stdout.trim().split("\n").at(-1) ?? "";
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(lastLine) as Record<string, unknown>;
  } catch {
    throw new Error(`Unsloth auth helper returned unexpected output: ${lastLine || "(empty)"}`);
  }
  const token = typeof parsed.token === "string" ? parsed.token : undefined;
  if (parsed.verified === true && token) {
    return { baseUrl, token, expiresAt: parseJwtExp(token) };
  }
  const reason = typeof parsed.reason === "string" && parsed.reason
    ? parsed.reason
    : "the server did not accept the desktop token flow";
  return { reason };
}

export function parseJwtExp(token: string): number | undefined {
  const parts = token.split(".");
  if (parts.length !== 3) return undefined;
  try {
    const payload = JSON.parse(
      Buffer.from(parts[1], "base64url").toString("utf8"),
    ) as Record<string, unknown>;
    const exp = payload.exp;
    return typeof exp === "number" && Number.isFinite(exp) ? exp : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Token cache keyed by base URL: a token minted for one server is never reused
 * for another, so changing UNSLOTH_STUDIO_URL mid-session forces a fresh mint
 * instead of sending a token to the wrong place.
 */
export class TokenCache {
  private cached: MintedToken | undefined;
  private mint: TokenMinter | undefined;
  private readonly now: () => number;

  constructor(
    mint: TokenMinter | undefined,
    now: () => number = () => Math.floor(Date.now() / 1000),
  ) {
    this.mint = mint;
    this.now = now;
  }

  setMinter(mint: TokenMinter | undefined): void {
    this.mint = mint;
  }

  invalidate(): void {
    this.cached = undefined;
  }

  async get(baseUrl: string): Promise<MintedToken | undefined> {
    const base = normalizeStudioUrl(baseUrl);
    const current = this.cached;
    const fresh =
      current !== undefined &&
      current.baseUrl === base &&
      (!current.expiresAt || current.expiresAt - this.now() > TOKEN_REFRESH_MARGIN_SECONDS);
    if (current && fresh) return current;
    if (!this.mint) return undefined;
    const minted = await this.mint();
    if (!minted) {
      this.cached = undefined;
      return undefined;
    }
    // Trust the JWT's own exp over whatever the minter reported.
    this.cached = { ...minted, expiresAt: minted.expiresAt ?? parseJwtExp(minted.token) };
    return this.cached;
  }
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

export interface ResolvedAuth {
  auth: { apiKey?: string; baseUrl: string };
  source: "api_key" | "env" | "minted";
}

export interface AuthResolutionInput {
  /** Server URL from a stored credential. */
  serverUrl?: string;
  /** Explicit API key from a stored credential. */
  apiKey?: string;
  env: Env;
  /** Override minting (tests, or a cached minter). */
  mint?: (baseUrl: string) => Promise<MintOutcome>;
}

/** Resolve credentials for the one server we have chosen to talk to. */
export async function resolveAuth(
  input: AuthResolutionInput,
): Promise<ResolvedAuth | undefined> {
  const target = resolveTarget(input);

  // 1. An explicit API key (stored credential or environment) works anywhere.
  const apiKey = input.apiKey ?? input.env[UNSLOTH_API_KEY_ENV];
  if (apiKey && apiKey.trim()) {
    return {
      auth: { apiKey: apiKey.trim(), baseUrl: target.baseUrl },
      source: input.apiKey ? "api_key" : "env",
    };
  }

  // 2. Otherwise self-issue a token for exactly this base URL.
  const mint =
    input.mint ?? ((baseUrl: string) => mintToken({ baseUrl, env: input.env }));
  const outcome = await mint(target.baseUrl);
  if (isMintedToken(outcome)) {
    return { auth: { apiKey: outcome.token, baseUrl: target.baseUrl }, source: "minted" };
  }
  // A configured server that refuses to hand out a token is an error worth
  // reporting; an unconfigured default that is simply not running is not.
  if (target.explicit && outcome && "reason" in outcome) {
    throw new Error(
      `Could not authenticate to ${target.baseUrl}: ${outcome.reason}. ` +
        `Start Unsloth Studio there, point ${UNSLOTH_STUDIO_URL_ENV} at the right port, ` +
        `or set ${UNSLOTH_API_KEY_ENV}.`,
    );
  }
  return undefined;
}
