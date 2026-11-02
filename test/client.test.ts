// SPDX-License-Identifier: MIT
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  UnslothClient,
  UnslothError,
  isConnectionError,
  normalizeStudioUrl,
} from "../extensions/unsloth/client.ts";
import {
  MODEL_LIST_PAYLOAD,
  STATUS_PAYLOAD,
  sendJson,
  startFakeServer,
  type FakeServer,
} from "../test-helpers/fake-server.ts";

async function withServer(
  handler: Parameters<typeof startFakeServer>[0],
  run: (server: FakeServer) => Promise<void>,
): Promise<void> {
  const server = await startFakeServer(handler);
  try {
    await run(server);
  } finally {
    await server.close();
  }
}

test("normalizeStudioUrl strips trailing slashes", () => {
  assert.equal(normalizeStudioUrl("http://127.0.0.1:8888/"), "http://127.0.0.1:8888");
  assert.equal(normalizeStudioUrl("http://127.0.0.1:8888///"), "http://127.0.0.1:8888");
  assert.equal(normalizeStudioUrl(" http://x:1 "), "http://x:1");
});

test("listModels parses the OpenAI-style catalog", async () => {
  await withServer((_req, res) => sendJson(res, 200, MODEL_LIST_PAYLOAD), async (server) => {
    const client = new UnslothClient({ baseUrl: server.baseUrl, apiKey: "k" });
    const entries = await client.listModels();
    assert.equal(entries.length, 3);
    assert.equal(entries[0].id, "unsloth/Qwen3.5-9B-GGUF");
    assert.equal(entries[0].loaded, true);
    assert.equal(entries[0].status.value, "loaded");
    assert.equal(entries[0].quant, "Q8_0");
    assert.equal(entries[0].contextLength, 262144);
    assert.equal(entries[1].loaded, false);
    assert.equal(entries[1].displayName, "flint-GGUF");
    assert.equal(entries[2].task, "text-to-image");
    const auth = server.requests[0]?.headers.authorization;
    assert.equal(auth, "Bearer k");
    assert.match(server.requests[0]?.headers["user-agent"] as string, /pi-unsloth/);
  });
});

test("listModels tolerates an empty data array", async () => {
  await withServer((_req, res) => sendJson(res, 200, { object: "list", data: [] }), async (server) => {
    const client = new UnslothClient({ baseUrl: server.baseUrl });
    assert.deepEqual(await client.listModels(), []);
  });
});

test("load sends model_path, variant and context and waits for the full padded body", async () => {
  await withServer(
    (_req, res, recorded) => {
      assert.equal(recorded.url, "/api/inference/load");
      const body = JSON.parse(recorded.body) as Record<string, unknown>;
      assert.equal(body.model_path, "unsloth/flint-GGUF");
      assert.equal(body.gguf_variant, "UD-Q4_K_XL");
      assert.equal(body.max_seq_length, 65536);
      assert.equal(recorded.headers.authorization, "Bearer k");
      sendJson(res, 200, { status: "loaded", model: "unsloth/flint-GGUF" });
    },
    async (server) => {
      const client = new UnslothClient({ baseUrl: server.baseUrl, apiKey: "k" });
      const result = await client.load({
        modelPath: "unsloth/flint-GGUF",
        ggufVariant: "UD-Q4_K_XL",
        maxSeqLength: 65536,
      });
      assert.equal(result.model, "unsloth/flint-GGUF");
    },
  );
});

test("load omits optional fields when unset", async () => {
  await withServer((_req, res, recorded) => {
    const body = JSON.parse(recorded.body) as Record<string, unknown>;
    assert.deepEqual(Object.keys(body), ["model_path"]);
    sendJson(res, 200, { ok: true });
  }, async (server) => {
    const client = new UnslothClient({ baseUrl: server.baseUrl });
    await client.load({ modelPath: "x" });
  });
});

test("load surfaces a deferred error from a padded 200 response", async () => {
  await withServer(
    (_req, res) =>
      sendJson(res, 200, {
        _deferred_error: { status_code: 507, detail: "out of VRAM" },
      }),
    async (server) => {
      const client = new UnslothClient({ baseUrl: server.baseUrl });
      await assert.rejects(
        client.load({ modelPath: "x" }),
        (error: unknown) =>
          error instanceof UnslothError &&
          error.message === "out of VRAM" &&
          error.status === 507,
      );
    },
  );
});

test("load rejects an empty padded body (truncated connection)", async () => {
  await withServer((_req, res) => {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end("");
  }, async (server) => {
    const client = new UnslothClient({ baseUrl: server.baseUrl });
    await assert.rejects(
      client.load({ modelPath: "x" }),
      /did not report completion/,
    );
  });
});

test("unload sends model_path", async () => {
  await withServer((_req, res, recorded) => {
    assert.equal(recorded.url, "/api/inference/unload");
    const body = JSON.parse(recorded.body) as Record<string, unknown>;
    assert.equal(body.model_path, "unsloth/Qwen3.5-9B-GGUF");
    sendJson(res, 200, { status: "unloaded", model: "unsloth/Qwen3.5-9B-GGUF" });
  }, async (server) => {
    const client = new UnslothClient({ baseUrl: server.baseUrl });
    const result = await client.unload({ modelPath: "unsloth/Qwen3.5-9B-GGUF" });
    assert.equal(result.status, "unloaded");
  });
});

test("HTTP errors carry the server detail message", async () => {
  await withServer((_req, res) => sendJson(res, 401, { detail: "Invalid token payload" }), async (server) => {
    const client = new UnslothClient({ baseUrl: server.baseUrl, apiKey: "bad" });
    await assert.rejects(
      client.listModels(),
      (error: unknown) =>
        error instanceof UnslothError &&
        error.message === "Invalid token payload" &&
        error.status === 401,
    );
  });
});

test("status parses the active-model report", async () => {
  await withServer((_req, res) => sendJson(res, 200, STATUS_PAYLOAD), async (server) => {
    const client = new UnslothClient({ baseUrl: server.baseUrl });
    const status = await client.status();
    assert.equal(status.modelIdentifier, "unsloth/Qwen3.5-9B-GGUF");
    assert.equal(status.ggufVariant, "Q8_0");
    assert.equal(status.contextLength, 262144);
    assert.equal(status.supportsReasoning, true);
    assert.equal(status.isVision, true);
  });
});

test("ggufVariants parses quant options", async () => {
  await withServer(
    (_req, res, recorded) => {
      assert.match(recorded.url ?? "", /repo_id=unsloth%2Fflint-GGUF/);
      sendJson(res, 200, {
        repo_id: "unsloth/flint-GGUF",
        variants: [
          { filename: "flint-Q8_0.gguf", quant: "Q8_0", size_bytes: 42301076928, downloaded: true },
          { filename: "flint-UD-Q4_K_XL.gguf", quant: "UD-Q4_K_XL", size_bytes: 25054688704 },
        ],
      });
    },
    async (server) => {
      const client = new UnslothClient({ baseUrl: server.baseUrl });
      const variants = await client.ggufVariants("unsloth/flint-GGUF");
      assert.equal(variants.length, 2);
      assert.equal(variants[0].quant, "Q8_0");
      assert.equal(variants[0].sizeBytes, 42301076928);
      assert.equal(variants[0].downloaded, true);
      assert.equal(variants[1].downloaded, false);
    },
  );
});

test("isConnectionError classifies fetch failures", () => {
  assert.equal(isConnectionError(new TypeError("fetch failed")), true);
  assert.equal(isConnectionError(new Error("socket hang up")), false);
  assert.equal(isConnectionError(new UnslothError("Invalid token", 401)), false);
});
