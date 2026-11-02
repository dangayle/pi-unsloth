// SPDX-License-Identifier: MIT
// Authentication against a running Unsloth Studio backend.
//
// Reuses Unsloth's own CLI machinery: `unsloth_cli._inference` provides
// find_studio_server (server discovery + health probe),
// verify_studio_identity (HMAC proof that the loopback responder is really
// this machine's Unsloth, not a port squatter), and _studio_token (self-issued
// desktop JWT signed with the shared secret in studio.db). We shell out to the
// bundled `unsloth_studio` Python venv and reuse those functions verbatim
// instead of reimplementing their safety checks.
//
// A user API key (UNSLOTH_API_KEY, or one stored through /login unsloth)
// takes precedence and works against any server URL, including non-loopback.

import { spawn, type ChildProcess, type SpawnOptions } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { DEFAULT_STUDIO_URL, normalizeStudioUrl } from "./client.ts";

/** Test seam: a spawn-compatible function. */
export type SpawnFn = (
  command: string,
  args: readonly string[],
  options: SpawnOptions,
) => ChildProcess;

export const UNSLOTH_STUDIO_URL_ENV = "UNSLOTH_STUDIO_URL";
export const UNSLOTH_API_KEY_ENV = "UNSLOTH_API_KEY";

/** Margin (seconds) before a minted JWT's expiry at which we re-mint. */
const TOKEN_REFRESH_MARGIN_SECONDS = 300;

export interface MintedToken {
  base: string;
  token: string;
  /** JWT `exp` claim, in seconds since the epoch. */
  expiresAt?: number;
}

export type TokenMinter = () => Promise<MintedToken | undefined>;

export interface ResolvedAuth {
  auth: { apiKey?: string; baseUrl: string };
  source: "api_key" | "minted" | "env";
}

export interface AuthResolutionInput {
  /** Server URL, e.g. from UNSLOTH_STUDIO_URL or a stored credential. */
  serverUrl?: string;
  /** Explicit API key, e.g. from UNSLOTH_API_KEY or a stored credential. */
  apiKey?: string;
  env: Record<string, string | undefined>;
  mint?: TokenMinter;
}

export interface MintOptions {
  pythonPath?: string;
  env: Record<string, string | undefined>;
  spawnFn?: SpawnFn;
  signal?: AbortSignal;
  timeoutMs?: number;
}

/** Where the bundled `unsloth_studio` venv lives on this machine. */
export function studioHome(env: Record<string, string | undefined>): string {
  const configured =
    env.UNSLOTH_STUDIO_HOME ?? env.STUDIO_HOME ?? env.UNSLOTH_HOME;
  if (configured && configured.trim()) return configured.trim();
  return path.join(os.homedir(), ".unsloth", "studio");
}

/** The Python interpreter of the bundled unsloth_studio venv, if present. */
export function findStudioPython(env: Record<string, string | undefined>): string | undefined {
  const explicit = env.UNSLOTH_PYTHON;
  if (explicit && explicit.trim()) return explicit.trim();
  const home = studioHome(env);
  const venv = path.join(home, "unsloth_studio");
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

/**
 * One-shot: discover the Studio server, verify its identity, and self-issue a
 * desktop token — all through unsloth_cli's own functions.
 */
export async function mintToken(options: MintOptions): Promise<MintedToken | undefined> {
  const pythonPath = options.pythonPath ?? findStudioPython(options.env);
  if (!pythonPath) return undefined;
  const site = sitePackagesOf(pythonPath);
  const home = studioHome(options.env);
  const script = [
    "import json, sys",
    `sys.path.insert(0, ${JSON.stringify(site)})`,
    `sys.path.insert(0, ${JSON.stringify(home)})`,
    "from unsloth_cli._inference import find_studio_server, verify_studio_identity, _studio_token, is_loopback_url",
    "base = find_studio_server()",
    "result = {\"base\": base}",
    "if base and is_loopback_url(base) and verify_studio_identity(base):",
    "    token = _studio_token()",
    "    if token:",
    "        result[\"token\"] = token",
    "print(json.dumps(result))",
  ].join("\n");
  const spawnFn = options.spawnFn ?? spawn;
  const child = spawnFn(pythonPath, ["-c", script], {
    cwd: site,
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
    throw new Error(
      `Unsloth auth helper failed: ${detail || `exit code ${exitCode}`}`,
    );
  }
  const lastLine = stdout.trim().split("\n").at(-1) ?? "";
  try {
    const parsed = JSON.parse(lastLine) as Record<string, unknown>;
    const base = typeof parsed.base === "string" ? parsed.base : undefined;
    const token = typeof parsed.token === "string" ? parsed.token : undefined;
    if (!base || !token) return undefined;
    return { base, token, expiresAt: parseJwtExp(token) };
  } catch {
    throw new Error(`Unsloth auth helper returned unexpected output: ${lastLine || "(empty)"}`);
  }
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
 * Cached token minter: reuses the minted token until it is close to expiring.
 * A mint involves a Python subprocess and an HMAC round trip, so it is not
 * something to run per request.
 */
export class TokenCache {
  private token: MintedToken | undefined;
  private mint: TokenMinter | undefined;
  private readonly now: () => number;

  constructor(mint: TokenMinter | undefined, now: () => number = () => Math.floor(Date.now() / 1000)) {
    this.mint = mint;
    this.now = now;
  }

  setMinter(mint: TokenMinter | undefined): void {
    this.mint = mint;
  }

  invalidate(): void {
    this.token = undefined;
  }

  async get(): Promise<MintedToken | undefined> {
    const current = this.token;
    if (current && (!current.expiresAt || current.expiresAt - this.now() > TOKEN_REFRESH_MARGIN_SECONDS)) {
      return current;
    }
    if (!this.mint) return undefined;
    const minted = await this.mint();
    if (minted) {
      // Derive the expiry from the JWT itself so the cache never depends on
      // the minter remembering to fill in expiresAt.
      const stored = {
        ...minted,
        expiresAt: minted.expiresAt ?? parseJwtExp(minted.token),
      };
      this.token = stored;
      return stored;
    }
    return undefined;
  }
}

/** Resolve usable credentials for the Unsloth provider. */
export async function resolveAuth(input: AuthResolutionInput): Promise<ResolvedAuth | undefined> {
  const env = input.env;
  const baseUrl = normalizeStudioUrl(input.serverUrl ?? env[UNSLOTH_STUDIO_URL_ENV] ?? DEFAULT_STUDIO_URL);

  // 1. An explicit API key (stored credential or environment) works everywhere.
  const apiKey = input.apiKey ?? env[UNSLOTH_API_KEY_ENV];
  if (apiKey && apiKey.trim()) {
    return {
      auth: { apiKey: apiKey.trim(), baseUrl },
      source: input.apiKey ? "api_key" : "env",
    };
  }

  // 2. Otherwise, self-issue a desktop JWT through Unsloth's own helpers.
  //    The helpers refuse anything but a verified loopback server, so a
  //    minted token is never sent to a non-local endpoint.
  const mint = input.mint ?? (() => mintToken({ env }));
  const minted = await mint();
  if (minted) {
    return {
      auth: { apiKey: minted.token, baseUrl: minted.base },
      source: "minted",
    };
  }
  return undefined;
}
