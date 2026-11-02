// SPDX-License-Identifier: MIT
import { test } from "node:test";
import assert from "node:assert/strict";
import type { ModelListEntry } from "../extensions/unsloth/client.ts";
import {
  findEntry,
  flagValue,
  numberFlag,
  tokenize,
} from "../extensions/unsloth/args.ts";

test("tokenize splits on whitespace and honors quotes", () => {
  assert.deepEqual(tokenize("load unsloth/flint-GGUF --variant UD-Q4_K_XL"), [
    "load",
    "unsloth/flint-GGUF",
    "--variant",
    "UD-Q4_K_XL",
  ]);
  assert.deepEqual(tokenize(""), []);
  assert.deepEqual(tokenize("  a   b "), ["a", "b"]);
  assert.deepEqual(tokenize(`load "a model/id" --variant 'Q8_0'`), [
    "load",
    "a model/id",
    "--variant",
    "Q8_0",
  ]);
  assert.deepEqual(tokenize(`unload "unclosed`), ["unload", "unclosed"]);
});

test("flagValue and numberFlag read option values", () => {
  const tokens = tokenize("load x --variant Q8_0 --context 65536");
  assert.equal(flagValue(tokens, "--variant"), "Q8_0");
  assert.equal(flagValue(tokens, "--missing"), undefined);
  assert.equal(numberFlag(tokens, "--context"), 65536);
  assert.equal(numberFlag(tokens, "--nope"), undefined);
  assert.equal(numberFlag(tokenize("load x --context nope"), "--context"), undefined);
  assert.equal(numberFlag(tokenize("load x --context 0"), "--context"), undefined);
});

const catalog: ModelListEntry[] = [
  { id: "unsloth/Qwen3.5-9B-GGUF", loaded: true, status: { value: "loaded" } },
  {
    id: "unsloth/flint-GGUF",
    loaded: false,
    status: { value: "unloaded" },
    displayName: "flint-GGUF",
  },
  { id: "org/Custom-Model", loaded: false, status: { value: "unloaded" } },
];

test("findEntry matches exact ids case-insensitively", () => {
  assert.equal(findEntry(catalog, "unsloth/Qwen3.5-9B-GGUF")?.id, "unsloth/Qwen3.5-9B-GGUF");
  assert.equal(findEntry(catalog, "unsloth/qwen3.5-9b-gguf")?.id, "unsloth/Qwen3.5-9B-GGUF");
});

test("findEntry matches display names and ownerless suffixes", () => {
  assert.equal(findEntry(catalog, "flint-GGUF")?.id, "unsloth/flint-GGUF");
  assert.equal(findEntry(catalog, "Qwen3.5-9B-GGUF")?.id, "unsloth/Qwen3.5-9B-GGUF");
});

test("findEntry falls back to a substring match", () => {
  assert.equal(findEntry(catalog, "Custom")?.id, "org/Custom-Model");
});

test("findEntry returns undefined when nothing matches", () => {
  assert.equal(findEntry(catalog, "nope"), undefined);
});
