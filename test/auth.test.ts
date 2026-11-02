// SPDX-License-Identifier: MIT
import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import * as path from "node:path";
import type { ChildProcess } from "node:child_process";
import {
  TokenCache,
  definedEnv,
  findStudioPython,
  isMintedToken,
  mintToken,
  parseJwtExp,
  probeStudio,
  resolveAuth,
  resolveTarget,
  studioHome,
  verifyAccess,
  type SpawnFn,
} from "../extensions/unsloth/auth.ts";
import { sendJson, startFakeServer } from "../test-helpers/fake-server.ts";

function tokenWithExp(exp: number): string {
  const header = Buffer.from(JSON.stringify({ alg: "HS256", typ: "JWT" })).toString("base64url");
  const payload = Buffer.from(
    JSON.stringify({ sub: "unsloth", desktop: true, exp }),
    "utf8",
  ).toString("base64url");
  return `${header}.${payload}.sig`;
}

interface FakeChildOptions {
  stdout?: string;
  stderr?: string;
  exitCode?: number;
  signal?: NodeJS.Signals | null;
  spawnError?: boolean;
}

function fakeChild(options: FakeChildOptions): ChildProcess {
  const child = new EventEmitter() as unknown as ChildProcess;
  const stdout = new PassThrough();
  const stderr = new PassThrough();
  (child as unknown as { stdout: PassThrough }).stdout = stdout;
  (child as unknown as { stderr: PassThrough }).stderr = stderr;
  queueMicrotask(() => {
    if (options.spawnError) {
      child.emit("error", new Error("spawn python ENOENT"));
      return;
    }
    stdout.end(options.stdout ?? "");
    stderr.end(options.stderr ?? "");
    child.emit("close", options.exitCode ?? 0, options.signal ?? null);
  });
  return child;
}

function spawnReturning(options: FakeChildOptions): SpawnFn {
  return () => fakeChild(options);
}

const VERIFIED = (exp: number) =>
  JSON.stringify({ base: "http://127.0.0.1:8899", verified: true, token: tokenWithExp(exp) });

// ---------------------------------------------------------------------------
// Target selection: we decide the port, not the server
// ---------------------------------------------------------------------------

test("resolveTarget prefers the credential, then the environment, then the default", () => {
  assert.deepEqual(
    resolveTarget({ serverUrl: "http://127.0.0.1:9001/", env: { UNSLOTH_STUDIO_URL: "http://127.0.0.1:9002" } }),
    { baseUrl: "http://127.0.0.1:9001", origin: "credential", explicit: true },
  );
  assert.deepEqual(
    resolveTarget({ env: { UNSLOTH_STUDIO_URL: "http://127.0.0.1:9002/" } }),
    { baseUrl: "http://127.0.0.1:9002", origin: "env", explicit: true },
  );
  assert.deepEqual(resolveTarget({ env: {} }), {
    baseUrl: "http://127.0.0.1:8888",
    origin: "default",
    explicit: false,
  });
});

test("resolveTarget ignores blank configuration", () => {
  assert.equal(resolveTarget({ serverUrl: "  ", env: { UNSLOTH_STUDIO_URL: "" } }).origin, "default");
});

// ---------------------------------------------------------------------------
// Probing / access checks
// ---------------------------------------------------------------------------

test("probeStudio recognizes an Unsloth backend", async () => {
  const server = await startFakeServer((_req, res) =>
    sendJson(res, 200, { status: "healthy", service: "Unsloth UI Backend", studio_root_id: "abc" }),
  );
  try {
    const probe = await probeStudio(server.baseUrl);
    assert.equal(probe.reachable, true);
    assert.equal(probe.isStudio, true);
    assert.equal(probe.detail, undefined);
  } finally {
    await server.close();
  }
});

test("probeStudio reports something that is not Unsloth", async () => {
  const server = await startFakeServer((_req, res) => sendJson(res, 200, { hello: "world" }));
  try {
    const probe = await probeStudio(server.baseUrl);
    assert.equal(probe.reachable, true);
    assert.equal(probe.isStudio, false);
    assert.match(probe.detail ?? "", /not an Unsloth Studio server/);
  } finally {
    await server.close();
  }
});

test("probeStudio names an unreachable port", async () => {
  const probe = await probeStudio("http://127.0.0.1:9", { timeoutMs: 2_000 });
  assert.equal(probe.reachable, false);
  assert.match(probe.detail ?? "", /nothing is answering at http:\/\/127\.0\.0\.1:9/);
});

test("verifyAccess accepts a working credential on the expected server", async () => {
  const server = await startFakeServer((_req, res, recorded) => {
    assert.equal(recorded.url, "/api/inference/loaded-models");
    assert.equal(recorded.headers.authorization, "Bearer good");
    sendJson(res, 200, { object: "list", data: [] });
  });
  try {
    await verifyAccess({ baseUrl: server.baseUrl, apiKey: "good" });
  } finally {
    await server.close();
  }
});

test("verifyAccess explains a rejected credential", async () => {
  const server = await startFakeServer((_req, res) => sendJson(res, 401, { detail: "Invalid token" }));
  try {
    await assert.rejects(
      verifyAccess({ baseUrl: server.baseUrl, apiKey: "bad" }),
      /rejected the credentials .*studio\.db differs/s,
    );
  } finally {
    await server.close();
  }
});

test("verifyAccess explains an unreachable server", async () => {
  await assert.rejects(
    verifyAccess({ baseUrl: "http://127.0.0.1:9", apiKey: "k", timeoutMs: 2_000 }),
    /Could not reach the Unsloth server at http:\/\/127\.0\.0\.1:9/,
  );
});

// ---------------------------------------------------------------------------
// Minting
// ---------------------------------------------------------------------------

test("parseJwtExp reads the exp claim", () => {
  assert.equal(parseJwtExp(tokenWithExp(1793600777)), 1793600777);
  assert.equal(parseJwtExp("not-a-jwt"), undefined);
  assert.equal(parseJwtExp("a.b.c"), undefined);
});

test("studioHome honors UNSLOTH_STUDIO_HOME and defaults to ~/.unsloth/studio", () => {
  assert.equal(studioHome({ UNSLOTH_STUDIO_HOME: "/custom" }), "/custom");
  assert.ok(studioHome({}).endsWith(path.join(".unsloth", "studio")));
});

test("findStudioPython prefers UNSLOTH_PYTHON and detects a missing venv", () => {
  assert.equal(findStudioPython({ UNSLOTH_PYTHON: "/opt/unsloth/python" }), "/opt/unsloth/python");
  assert.equal(
    findStudioPython({ UNSLOTH_STUDIO_HOME: path.join("test-helpers", "no-venv-here") }),
    undefined,
  );
});

test("definedEnv drops undefined values", () => {
  assert.deepEqual(definedEnv({ A: "1", B: undefined }), { A: "1" });
});

test("mintToken targets the requested base and never asks a helper to pick one", async () => {
  let script = "";
  let seenEnv: Record<string, string> | undefined;
  const result = await mintToken({
    baseUrl: "http://127.0.0.1:8899/",
    pythonPath: "/fake/python",
    env: { UNSLOTH_STUDIO_URL: "http://127.0.0.1:8899" },
    spawnFn: ((_cmd, args, options) => {
      script = args[1] ?? "";
      seenEnv = options.env as Record<string, string>;
      return fakeChild({ stdout: VERIFIED(1793600777) });
    }) as SpawnFn,
  });
  // The exact base is embedded, and discovery heuristics are not used at all.
  assert.match(script, /base = "http:\/\/127\.0\.0\.1:8899"/);
  assert.doesNotMatch(script, /find_studio_server/);
  assert.match(script, /verify_studio_identity/);
  assert.match(script, /_studio_token/);
  assert.equal(seenEnv?.UNSLOTH_STUDIO_URL, "http://127.0.0.1:8899");
  assert.ok(isMintedToken(result));
  assert.equal(result.baseUrl, "http://127.0.0.1:8899");
  assert.equal(result.expiresAt, 1793600777);
});

test("mintToken reports a verification failure as a reason", async () => {
  const result = await mintToken({
    baseUrl: "http://127.0.0.1:8899",
    pythonPath: "/fake/python",
    env: {},
    spawnFn: spawnReturning({
      stdout: JSON.stringify({ base: "http://127.0.0.1:8899", verified: false, reason: "the server at that address did not prove the shared secret from studio.db" }),
    }),
  });
  assert.ok(result && !isMintedToken(result));
  assert.match(result.reason, /shared secret/);
});

test("mintToken reports a helper failure with its stderr", async () => {
  await assert.rejects(
    mintToken({
      baseUrl: "http://127.0.0.1:8899",
      pythonPath: "/fake/python",
      env: {},
      spawnFn: spawnReturning({ stderr: "Traceback…\nRuntimeError: studio.db locked", exitCode: 1 }),
    }),
    /studio\.db locked/,
  );
});

test("mintToken returns undefined when python cannot be started", async () => {
  const result = await mintToken({
    baseUrl: "http://127.0.0.1:8899",
    pythonPath: "/missing/python",
    env: {},
    spawnFn: spawnReturning({ spawnError: true }),
  });
  assert.equal(result, undefined);
});

test("mintToken rejects unexpected helper output", async () => {
  await assert.rejects(
    mintToken({
      baseUrl: "http://127.0.0.1:8899",
      pythonPath: "/fake/python",
      env: {},
      spawnFn: spawnReturning({ stdout: "import torch failed" }),
    }),
    /unexpected output/,
  );
});

test("mintToken returns undefined for an aborted operation", async () => {
  const controller = new AbortController();
  controller.abort();
  const result = await mintToken({
    baseUrl: "http://127.0.0.1:8899",
    pythonPath: "/fake/python",
    env: {},
    signal: controller.signal,
    spawnFn: spawnReturning({ stdout: "{}" }),
  });
  assert.equal(result, undefined);
});

// ---------------------------------------------------------------------------
// Token cache
// ---------------------------------------------------------------------------

test("TokenCache reuses an unexpired token for the same base", async () => {
  let calls = 0;
  const cache = new TokenCache(async () => {
    calls += 1;
    return { baseUrl: "http://127.0.0.1:8899", token: tokenWithExp(1000) };
  }, () => 100);
  await cache.get("http://127.0.0.1:8899");
  await cache.get("http://127.0.0.1:8899");
  assert.equal(calls, 1);
});

test("TokenCache re-mints when the base URL changes", async () => {
  let calls = 0;
  const cache = new TokenCache(async () => {
    calls += 1;
    return { baseUrl: "http://127.0.0.1:8899", token: tokenWithExp(1000) };
  }, () => 100);
  await cache.get("http://127.0.0.1:8899");
  await cache.get("http://127.0.0.1:9000");
  // A token minted for one server must never be sent to another.
  assert.equal(calls, 2);
});

test("TokenCache re-mints near expiry and after invalidation", async () => {
  let calls = 0;
  let now = 100;
  const cache = new TokenCache(async () => {
    calls += 1;
    return { baseUrl: "http://127.0.0.1:8899", token: tokenWithExp(now + 3600) };
  }, () => now);
  await cache.get("http://127.0.0.1:8899");
  now = 100 + 3600 - 250;
  await cache.get("http://127.0.0.1:8899");
  assert.equal(calls, 2);
  cache.invalidate();
  await cache.get("http://127.0.0.1:8899");
  assert.equal(calls, 3);
});

test("TokenCache derives expiry from the JWT when the minter omits it", async () => {
  const cache = new TokenCache(
    async () => ({ baseUrl: "http://127.0.0.1:8899", token: tokenWithExp(4242) }),
    () => 100,
  );
  const minted = await cache.get("http://127.0.0.1:8899");
  assert.equal(minted?.expiresAt, 4242);
});

test("TokenCache without a minter returns undefined", async () => {
  const cache = new TokenCache(undefined);
  assert.equal(await cache.get("http://127.0.0.1:8899"), undefined);
});

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

test("resolveAuth prefers a stored API key, then the environment, then minting", async () => {
  const env: Record<string, string | undefined> = {};
  const mint = async () => ({ baseUrl: "http://127.0.0.1:8888", token: "minted-token" });

  const stored = await resolveAuth({ apiKey: "stored-key", env, mint });
  assert.deepEqual(stored, {
    auth: { apiKey: "stored-key", baseUrl: "http://127.0.0.1:8888" },
    source: "api_key",
  });

  const fromEnv = await resolveAuth({ env: { UNSLOTH_API_KEY: "env-key" }, mint });
  assert.equal(fromEnv?.auth.apiKey, "env-key");
  assert.equal(fromEnv?.source, "env");

  const minted = await resolveAuth({ env, mint });
  assert.deepEqual(minted, {
    auth: { apiKey: "minted-token", baseUrl: "http://127.0.0.1:8888" },
    source: "minted",
  });
});

test("resolveAuth keeps the requested base URL even if the minter reports another", async () => {
  const resolved = await resolveAuth({
    serverUrl: "http://127.0.0.1:8899",
    env: {},
    // A minter that answers with the wrong port must not redirect requests.
    mint: async () => ({ baseUrl: "http://127.0.0.1:8888", token: "t" }),
  });
  assert.equal(resolved?.auth.baseUrl, "http://127.0.0.1:8899");
});

test("resolveAuth passes the exact requested base to the minter", async () => {
  const seen: string[] = [];
  await resolveAuth({
    env: { UNSLOTH_STUDIO_URL: "http://127.0.0.1:8899/" },
    mint: async (baseUrl) => {
      seen.push(baseUrl);
      return undefined;
    },
  });
  assert.deepEqual(seen, ["http://127.0.0.1:8899"]);
});

test("resolveAuth is loud when an explicitly configured server cannot mint", async () => {
  await assert.rejects(
    resolveAuth({
      env: { UNSLOTH_STUDIO_URL: "http://127.0.0.1:8899" },
      mint: async () => ({ reason: "the server at that address did not prove the shared secret from studio.db" }),
    }),
    /Could not authenticate to http:\/\/127\.0\.0\.1:8899.*UNSLOTH_STUDIO_URL/s,
  );
});

test("resolveAuth stays quiet when the default server is simply not running", async () => {
  const resolved = await resolveAuth({
    env: {},
    mint: async () => ({ reason: "nothing there" }),
  });
  assert.equal(resolved, undefined);
});

test("resolveAuth returns undefined without any usable credential", async () => {
  assert.equal(await resolveAuth({ env: {}, mint: async () => undefined }), undefined);
});
