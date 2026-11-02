// SPDX-License-Identifier: MIT
// Regression tests for server selection: with more than one Unsloth server
// running, the plugin must talk to the port it was told to use and nowhere
// else. This is what went wrong when discovery was delegated to
// unsloth_cli.find_studio_server(), which prefers "the server this install
// recorded" and can answer with a port nobody asked about.

import { test } from "node:test";
import assert from "node:assert/strict";
import { UnslothClient } from "../extensions/unsloth/client.ts";
import { resolveAuth, resolveTarget, verifyAccess } from "../extensions/unsloth/auth.ts";
import { MODEL_LIST_PAYLOAD, sendJson, startFakeServer } from "../test-helpers/fake-server.ts";

test("with two servers running, only the requested one is contacted", async () => {
  const first = await startFakeServer((_req, res) => sendJson(res, 200, MODEL_LIST_PAYLOAD));
  const second = await startFakeServer((_req, res) => sendJson(res, 200, MODEL_LIST_PAYLOAD));
  try {
    const resolved = await resolveAuth({
      // The user pointed at the second server.
      serverUrl: second.baseUrl,
      apiKey: "key-for-second",
      env: {},
    });
    assert.ok(resolved);
    assert.equal(resolved.auth.baseUrl, second.baseUrl);

    const client = new UnslothClient({
      baseUrl: resolved.auth.baseUrl,
      apiKey: resolved.auth.apiKey,
    });
    const models = await client.listModels();
    assert.ok(models.length > 0);

    assert.equal(second.requests.length, 1, "the requested server must be the one asked");
    assert.equal(first.requests.length, 0, "the other server must never be contacted");
  } finally {
    await first.close();
    await second.close();
  }
});

test("a token minted for one port is checked against that port, not another", async () => {
  const first = await startFakeServer((_req, res) =>
    sendJson(res, 200, { status: "healthy", service: "Unsloth UI Backend", studio_root_id: "x" }),
  );
  const second = await startFakeServer((_req, res, recorded) => {
    if (recorded.url === "/api/inference/loaded-models") {
      sendJson(res, 200, { object: "list", data: [] });
      return;
    }
    sendJson(res, 200, { status: "healthy", service: "Unsloth UI Backend", studio_root_id: "y" });
  });
  try {
    await verifyAccess({ baseUrl: second.baseUrl, apiKey: "t" });
    assert.equal(second.requests.length, 1);
    assert.equal(first.requests.length, 0);
  } finally {
    await first.close();
    await second.close();
  }
});

test("a rejecting server fails loudly instead of silently using another", async () => {
  const wrong = await startFakeServer((_req, res) => sendJson(res, 401, { detail: "Invalid token" }));
  const right = await startFakeServer((_req, res) => sendJson(res, 200, { object: "list", data: [] }));
  try {
    // Requesting the wrong server must fail, not quietly fall through to a
    // server that would have accepted the request.
    await assert.rejects(verifyAccess({ baseUrl: wrong.baseUrl, apiKey: "t" }), /rejected the credentials/);
    assert.equal(right.requests.length, 0);
  } finally {
    await wrong.close();
    await right.close();
  }
});

test("resolveTarget follows UNSLOTH_STUDIO_URL exactly", async () => {
  const server = await startFakeServer((_req, res) => sendJson(res, 200, MODEL_LIST_PAYLOAD));
  try {
    const target = resolveTarget({ env: { UNSLOTH_STUDIO_URL: `${server.baseUrl}/` } });
    assert.equal(target.baseUrl, server.baseUrl);
    assert.equal(target.explicit, true);
    await new UnslothClient({ baseUrl: target.baseUrl }).listModels();
    assert.equal(server.requests.length, 1);
  } finally {
    await server.close();
  }
});
