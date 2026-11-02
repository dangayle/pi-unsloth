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
  mintToken,
  parseJwtExp,
  resolveAuth,
  studioHome,
  type SpawnFn,
} from "../extensions/unsloth/auth.ts";

function tokenWithExp(exp: number): string {
  const header = Buffer.from(JSON.stringify({ alg: "HS256", typ: "JWT" })).toString("base64url");
  const payload = Buffer.from(JSON.stringify({ sub: "unsloth", desktop: true, exp })).toString("base64url");
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

test("parseJwtExp reads the exp claim", () => {
  assert.equal(parseJwtExp(tokenWithExp(1793600777)), 1793600777);
  assert.equal(parseJwtExp("not-a-jwt"), undefined);
  assert.equal(parseJwtExp("a.b.c"), undefined);
});

test("studioHome honors UNSLOTH_STUDIO_HOME and defaults to ~/.unsloth/studio", () => {
  assert.equal(
    studioHome({ UNSLOTH_STUDIO_HOME: "/custom" }),
    "/custom",
  );
  const fallback = studioHome({});
  assert.ok(fallback.endsWith(path.join(".unsloth", "studio")));
});

test("findStudioPython prefers UNSLOTH_PYTHON and detects a missing venv", () => {
  assert.equal(findStudioPython({ UNSLOTH_PYTHON: "/opt/unsloth/python" }), "/opt/unsloth/python");
  // A studio home without a venv yields nothing.
  const missing = findStudioPython({ UNSLOTH_STUDIO_HOME: path.join("test-helpers", "no-venv-here") });
  assert.equal(missing, undefined);
});

test("TokenCache mints once and reuses an unexpired token", async () => {
  let calls = 0;
  const cache = new TokenCache(async () => {
    calls += 1;
    return { base: "http://x:8888", token: tokenWithExp(1000) };
  }, () => 100);
  const first = await cache.get();
  const second = await cache.get();
  assert.equal(calls, 1);
  assert.equal(second, first);
});

test("TokenCache re-mints near expiry and after invalidation", async () => {
  let calls = 0;
  let currentTime = 100;
  const cache = new TokenCache(
    async () => {
      calls += 1;
      return { base: "http://x:8888", token: tokenWithExp(currentTime + 3600) };
    },
    () => currentTime,
  );
  await cache.get();
  // Jump to within the refresh margin: expiry - now = 400s < 300s margin? No: 3600 - 900 = 2700. Walk closer.
  currentTime = 100 + 3600 - 250; // 250s left
  await cache.get();
  assert.equal(calls, 2);
  cache.invalidate();
  await cache.get();
  assert.equal(calls, 3);
});

test("TokenCache without a minter returns undefined", async () => {
  const cache = new TokenCache(undefined);
  assert.equal(await cache.get(), undefined);
});

test("mintToken reuses the unsloth_cli helpers and parses their output", async () => {
  let command = "";
  let script = "";
  const minted = await mintToken({
    pythonPath: "/fake/python",
    env: {},
    spawnFn: ((cmd, args, options) => {
      command = cmd;
      script = args[1] ?? "";
      void options;
      return fakeChild({
        stdout: JSON.stringify({ base: "http://127.0.0.1:8888", token: tokenWithExp(1793600777) }),
      });
    }) as SpawnFn,
  });
  assert.equal(command, "/fake/python");
  assert.match(script, /find_studio_server/);
  assert.match(script, /verify_studio_identity/);
  assert.match(script, /_studio_token/);
  assert.equal(minted?.base, "http://127.0.0.1:8888");
  assert.equal(minted?.expiresAt, 1793600777);
});

test("mintToken reports a helper failure with its stderr", async () => {
  await assert.rejects(
    mintToken({
      pythonPath: "/fake/python",
      env: {},
      spawnFn: spawnReturning({ stderr: "Traceback…\nRuntimeError: studio.db locked", exitCode: 1 }),
    }),
    /studio\.db locked/,
  );
});

test("mintToken returns undefined when python cannot be started", async () => {
  const result = await mintToken({
    pythonPath: "/missing/python",
    env: {},
    spawnFn: spawnReturning({ spawnError: true }),
  });
  assert.equal(result, undefined);
});

test("mintToken rejects unexpected helper output", async () => {
  await assert.rejects(
    mintToken({
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
    pythonPath: "/fake/python",
    env: {},
    signal: controller.signal,
    spawnFn: spawnReturning({ stdout: "{}" }),
  });
  assert.equal(result, undefined);
});

test("mintToken passes its environment to the helper process", async () => {
  let seenEnv: Record<string, string> | undefined;
  await mintToken({
    pythonPath: "/fake/python",
    env: { UNSLOTH_STUDIO_URL: "http://127.0.0.1:8899", UNSET_VALUE: undefined },
    spawnFn: ((cmd, args, options) => {
      seenEnv = options.env as Record<string, string>;
      void cmd;
      void args;
      return fakeChild({ stdout: JSON.stringify({ base: "http://127.0.0.1:8899", token: "t" }) });
    }) as SpawnFn,
  });
  // The helper reads UNSLOTH_STUDIO_URL, so a non-default port must reach it.
  assert.equal(seenEnv?.UNSLOTH_STUDIO_URL, "http://127.0.0.1:8899");
  assert.equal("UNSET_VALUE" in (seenEnv ?? {}), false);
});

test("definedEnv drops undefined values", () => {
  assert.deepEqual(definedEnv({ A: "1", B: undefined }), { A: "1" });
});

test("resolveAuth prefers a stored API key, then the environment, then minting", async () => {
  const env: Record<string, string | undefined> = {};
  const mint = async () => ({ base: "http://127.0.0.1:8888", token: "minted-token" });

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

test("resolveAuth normalizes the configured server URL", async () => {
  const resolved = await resolveAuth({
    apiKey: "k",
    serverUrl: "http://127.0.0.1:9999/",
    env: {},
    mint: async () => undefined,
  });
  assert.equal(resolved?.auth.baseUrl, "http://127.0.0.1:9999");
});

test("resolveAuth returns undefined without any usable credential", async () => {
  const resolved = await resolveAuth({
    env: {},
    mint: async () => undefined,
  });
  assert.equal(resolved, undefined);
});
